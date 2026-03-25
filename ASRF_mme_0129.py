import argparse
import re
import time
import torch
torch.backends.cuda.max_split_size_mb = 128
torch.cuda.empty_cache()
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
from tqdm import tqdm
# 分布式工具
import torch.distributed  as dist
import sys
import math
from PIL import Image
import requests
from io import BytesIO
import torch.nn.functional as F
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入LLaVA组件
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
# MemVR 安全版
# from my_final_mem_lla import apply_memvr_to_model, MemVRTracker, MemVRLlamaMLP, layer_history
from mem_simplified import apply_memvr_to_model, MemVRTracker, MemVRLlamaMLP, layer_history
# 分布式 & 日志
from utils import dist_util
from utils.logger import create_logger
from transformers import set_seed

# ED模块
from my_ed_qa import EDGenerationWrapper, EDProcessor
# ED模块
# from my_final_ed_lla import EDGenerationWrapper, EDProcessor
# utils/model_loader.py
import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def load_model_once(model_path, model_base=None, device='cuda'):
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, "llava_v1", device=device
    )
    if isinstance(model, (tuple, list)):
        model = model[0]
    model.half()
    torch.save(model.state_dict(), "/tmp/model_state.pt")
    return tokenizer, model, image_processor



# ==============================
# MME 风格 Yes/No 解析
# ==============================
def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]
    out = out.replace('.', '').replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


# =====================================================
# 图像加载
# =====================================================
def load_image(image_file):
    if image_file.startswith("http"):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


class DeCoMemVREDGenerator:
    """DeCo + MemVR + ED 三方法融合生成器 - 完整版"""

    def __init__(self, args, model, tokenizer, image_processor):
        self.args = args
        self.model = model.half().cuda()
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # 性能监控
        self.generation_times = []
        self.cache_hits = 0
        self.vision_features = None
        self.tracker = MemVRTracker()

        # =========================
        # DeCo配置
        # =========================
        self.use_deco = args.use_deco
        self.deco_start_layer = args.start_layer
        self.deco_end_layer = args.end_layer
        self.deco_alpha = args.alpha
        self.deco_top_p = args.threshold_top_p
        self.deco_top_k = args.threshold_top_k
        self.top_k = args.top_k


        # =========================
        # ED配置
        # =========================
        if args.use_ed:
            self.ed_config = {
                'ed_alpha': args.ed_alpha,
                'ed_beta': args.ed_beta,
                'ed_tau': args.ed_tau,
                'crop_size': args.ed_crop_size,
                'num_regions': 4,
                'use_ed': True
            }
            self.ed_processor = EDProcessor(model, tokenizer, image_processor, self.ed_config)
            self.ed_wrapper = EDGenerationWrapper(model, tokenizer, self.ed_processor)
        else:
            self.ed_config = None
            self.ed_processor = None
            self.ed_wrapper = None

        # 缓存
        self.input_cache = {}

    def cleanup_after_image(self):
        """🔥 每张图片结束后，彻底清理所有状态，防 OOM"""

        # ========= 1. 清理 MemVR 模块状态 =========
        for m in self.model.modules():
            if isinstance(m, MemVRLlamaMLP):
                m._visual_features = None
                m.cached_adapter = None
                m.cached_visual_feats = None
                m.entropy_history = []
                m._last_dynamic_layers = []
                m._printed_this_image = False
                m._second_pass = False
                m.first_pass_done = False
                m.second_pass_done = False

        # ========= 2. 清理 ED =========
        if self.ed_wrapper is not None:
            self.ed_wrapper.subimages = None
            if hasattr(self.ed_wrapper, "cached_logits"):
                self.ed_wrapper.cached_logits = None

        # ========= 3. 清理 Vision 特征 =========
        self.vision_features = None

        # ========= 4. 清理 Python 强引用 =========
        import gc
        gc.collect()

        # ========= 5. CUDA =========
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def prepare_inputs(self, image_file, question, img_id=None):
        """准备输入"""
        # 清空上一张图状态
        global memvr_sample_retraced
        memvr_sample_retraced = False  # ✅ 每个 question 只 reset 一次
        layer_history.clear()
        for m in self.model.modules():
            if isinstance(m, MemVRLlamaMLP):
                m._visual_features = None
                m._has_applied_memvr = False
                m.cached_adapter = None
                m.cached_visual_feats = None
                m._adapter_initialized = False
                m._printed_this_image = False
                m._second_pass = False
                m._last_dynamic_layers = []
                m._current_image_id = img_id

                # ✅ ★★★ 必须加的 ★★★
                m.first_pass_done = False
                m.second_pass_done = False

        # 缓存
        cache_key = f"{image_file}_{question[:50]}"
        if cache_key in self.input_cache:
            self.cache_hits += 1
            return self.input_cache[cache_key]

        # 1️⃣ 加载图像
        image = self.load_image(image_file)
        # 创建对话
        if "llama-2" in self.args.model_path.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.args.model_path.lower():
            conv_mode = "llava_v1"
        else:
            conv_mode = "llava_v0"

        # 处理问题
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        # 2️⃣ 构造 prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        # conv.append_message(conv.roles[0], qs + " Please answer this question with 'Yes' or 'No'.")
        # conv.append_message(conv.roles[0], qs + " Analyze the image carefully and answer 'Yes' or 'No' strictly based on visual evidence.")
        # " Analyze the image carefully and answer 'Yes' or 'No' strictly based on visual evidence. "
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        # 3️⃣ 图像预处理
        try:
            image_tensor = self.image_processor.preprocess(
                image, return_tensors="pt"
            )["pixel_values"].half().cuda()
        except Exception as e:
            print(f"[WARNING] 图像预处理失败: {e}")
            image_tensor = None

        # 4️⃣ MemVR注入
        if self.args.apply_memvr and image_tensor is not None:
            torch.cuda.empty_cache()
            vision_tower = self.model.get_vision_tower()
            feats_list = []
            B = image_tensor.shape[0]
            for i in range(0, B, 2):
                end = min(i + 2, B)
                imgs = image_tensor[i:end]
                with torch.no_grad():
                    f = vision_tower(imgs)
                    if hasattr(f, "last_hidden_state"):
                        f = f.last_hidden_state
                    elif isinstance(f, (tuple, list)):
                        f = f[0]
                    if f.ndim == 4:
                        Bc, C, H, W = f.shape
                        f = f.view(Bc, C, H * W).permute(0, 2, 1)
                    feats_list.append(f.half())
            feats = torch.cat(feats_list, dim=0)
            mean = feats.mean(dim=1, keepdim=True)
            std = feats.std(dim=1, keepdim=True) + 1e-6
            feats = ((feats - mean) / std * 3.0).tanh()
            self.vision_features = feats
            torch.cuda.empty_cache()

            # self.model, self.tracker = apply_memvr_to_model(
            #     self.model,
            #     tracker=self.tracker,
            #     config={
            #         "apply_memvr": True,
            #         "retracing_ratio": self.args.memvr_retracing_ratio,
            #         "memvr_backward_layers": self.args.memvr_backward_layers,
            #         "entropy_threshold_coef": 0.7,
            #     }
            # )

            for m in self.model.modules():
                if isinstance(m, MemVRLlamaMLP):
                    m.set_vision_features(self.vision_features)
                    m.tracker = self.tracker

        # 5️⃣ ED处理
        if self.args.use_ed and image_tensor is not None and self.ed_wrapper is not None:
            self.ed_wrapper.set_subimages(image_tensor)
            image_tensor = self._get_ed_tensor_safe()

        stopping_criteria = [KeywordsStoppingCriteria([conv.sep], self.tokenizer, input_ids)]
        result = (input_ids, image_tensor, stopping_criteria, full_prompt)
        self.input_cache[cache_key] = result
        return result

    def _get_ed_tensor_safe(self):
        if self.ed_wrapper is None or self.ed_wrapper.subimages is None:
            return None
        return self.ed_wrapper.subimages

    @staticmethod
    def load_image(image_file):
        if image_file.startswith("http"):
            response = requests.get(image_file)
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(image_file).convert("RGB")

    def generate(self, input_ids, image_tensor, stopping_criteria=None):
        """生成文本"""
        start_time = time.time()
        B = input_ids.shape[0]

        handles = []
        total_layers = sum(1 for m in self.model.modules() if isinstance(m, MemVRLlamaMLP))

        for m in self.model.modules():
            if isinstance(m, MemVRLlamaMLP):
                m.apply_memvr = m.layer_idx >= total_layers - self.args.memvr_backward_layers
                m._printed_this_image = False

                def hook(module, input, output):
                    if module.apply_memvr and not module._printed_this_image:
                        # entropy = module.entropy_history[-1] if module.entropy_history else 0.0
                        module._printed_this_image = True
                    return output

                handles.append(m.register_forward_hook(hook))

        # 生成参数
        gen_kwargs = {
            "input_ids": input_ids,
            "images": image_tensor,
            "do_sample": self.args.temperature > 0,
            # "do_sample": False,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "num_beams": getattr(self.args, "num_beams", 1),
            "max_new_tokens": min(self.args.max_new_tokens, 5),
            "return_dict_in_generate": True,
            "stopping_criteria": stopping_criteria,
        }

        output_text = ""

        try:
            # =========================
            # DeCo生成
            # =========================
            if self.use_deco:
                gen_kwargs.update({
                    "alpha": self.deco_alpha,
                    "threshold_top_p": self.deco_top_p,
                    "threshold_top_k": self.deco_top_k,
                    "early_exit_layers": list(range(self.deco_start_layer, self.deco_end_layer + 1))
                })
                print(f"[DeCo] alpha={self.deco_alpha}, 层范围: {self.deco_start_layer}-{self.deco_end_layer}")

            # =========================
            # ED生成
            # =========================
            if self.args.use_ed and self.ed_wrapper is not None:
                with torch.inference_mode():
                    try:
                        sequences = self.ed_wrapper.ed_generate_simple(
                            input_ids=input_ids,
                            images=image_tensor,
                            do_sample=self.args.temperature > 0,
                            temperature=self.args.temperature,
                            top_p=self.args.top_p,
                            max_new_tokens=min(self.args.max_new_tokens, 5),
                            stopping_criteria=stopping_criteria,
                            use_ed=True
                        )
                        input_len = input_ids.shape[1]
                        output_text = self.tokenizer.batch_decode(sequences[:, input_len:], skip_special_tokens=True)[0]

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"[ED生成失败: CUDA OOM] 重新启动脚本...")
                            import traceback
                            traceback.print_exc()
                            # 清理显存
                            torch.cuda.empty_cache()
                            # 重启当前脚本
                            import sys, os
                            os.execv(sys.executable, [sys.executable] + sys.argv)
                        else:
                            print(f"[ED生成失败] {e}, 回退标准生成")
                            import traceback
                            traceback.print_exc()
                            with torch.no_grad():
                                sequences = self.model.generate(**gen_kwargs)
                                input_len = input_ids.shape[1]
                                output_text = self.tokenizer.batch_decode(sequences.sequences[:, input_len:],
                                                                          skip_special_tokens=True)[0]
            else:
                # 标准生成（DeCo + MemVR）
                with torch.inference_mode():
                    with torch.no_grad():
                        sequences = self.model.generate(**gen_kwargs)
                    input_len = input_ids.shape[1]
                    output_text = \
                    self.tokenizer.batch_decode(sequences.sequences[:, input_len:], skip_special_tokens=True)[0]

        finally:
            for h in handles: h.remove()
            if self.vision_features is not None:
                del self.vision_features
                self.vision_features = None
            torch.cuda.empty_cache()

        self.generation_times.append(time.time() - start_time)
        return output_text.strip()

    def print_stats(self):
        if len(self.generation_times) > 0:
            avg_time = sum(self.generation_times) / len(self.generation_times)
            print(f"\n=== 性能统计 ===")
            print(f"平均生成时间: {avg_time:.2f}s")
            print(f"总生成次数: {len(self.generation_times)}")
            print(f"缓存命中: {self.cache_hits}")
            print(f"最长生成时间: {max(self.generation_times):.2f}s")
            print(f"最短生成时间: {min(self.generation_times):.2f}s")


# =========================
# 主评估函数
# =========================
def eval_model(args):
    dist_util.setup_dist(args)
    device = dist_util.device()

    if dist.get_rank() == 0:
        os.makedirs(args.log_path, exist_ok=True)
        logger = create_logger(args.log_path)
        logger.info(f"实验目录创建于 {args.log_path}")
    else:
        logger = create_logger(None)

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        get_model_name_from_path(args.model_path),
        device="cuda"
    )
    if isinstance(model, (tuple, list)):
        model = model[0]
    model.half()

    # =========================
    # MemVR 注入（保留）
    # =========================
    if args.apply_memvr:
        model, _ = apply_memvr_to_model(
            model,
            tracker=MemVRTracker(),
            config={
                "apply_memvr": True,
                "retracing_ratio": args.memvr_retracing_ratio,
                "memvr_backward_layers": args.memvr_backward_layers,
                "entropy_threshold_coef": args.entropy_threshold_coef,
                "first_pass_only_last_n":args.memvr_backward_layers
            }
        )

    generator = DeCoMemVREDGenerator(args, model, tokenizer, image_processor)

    # =====================================================
    # ★ POPE：读取问题文件（替换原 ours.jsonl）
    # =====================================================
    with open(args.question_file, "r") as f:
        lines = f.readlines()

    # if args.max_samples > 0:
    #     lines = lines[:args.max_samples]

    # =====================================================
    # Resume 机制（保留）
    # 这里用 question_id 作为断点标识
    # =====================================================
    # -----------------------------
    # Resume 精准逻辑
    # -----------------------------
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    last_done_qid = None
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for l in f:
                try:
                    obj = json.loads(l)
                    last_done_qid = obj["question_id"]
                except:
                    continue

    if last_done_qid is not None:
        print(f"[Resume] 上次完成 question_id={last_done_qid}")
        resume_flag = False
    else:
        print("[Resume] 无已完成记录，从头开始")
        resume_flag = True

    remaining_lines = []
    for l in lines:
        qid = json.loads(l)["question_id"]
        if resume_flag or qid > last_done_qid:
            remaining_lines.append(l)
            resume_flag = True  # 开始记录之后的所有题都要处理

    print(f"[Resume] 剩余待处理 {len(remaining_lines)} / {len(lines)}")

    # =====================================================
    # ★ 主循环（POPE）
    # =====================================================
    for data_line in tqdm(remaining_lines, desc="MME Eval", dynamic_ncols=True):
        torch.cuda.empty_cache()

        line = json.loads(data_line)
        question_id = line["question_id"]
        # 构建图像路径（COCO格式）
        image = line["image"]
        image_file = os.path.join(args.image_folder, line["image"])
        question = line["text"]

        try:
            input_ids, image_tensor, stopping_criteria, prompt = generator.prepare_inputs(
                image_file=image_file,
                question=question,
                img_id=question_id
            )

            output_text = generator.generate(
                input_ids,
                image_tensor,
                stopping_criteria
            )
            print(output_text)

            answer = recorder(output_text)
            print(answer)


            res_dict = {
                "question_id": question_id,
                "image": image_file,
                "prompt": question,
                "text": answer,
                "model_id": args.model_path,
                "metadata": {}
            }

            with open(answers_file, "a") as ans_file:
                ans_file.write(json.dumps(res_dict, ensure_ascii=False) + "\n")

            # ===== 一题结束，显存保护（保留）=====
            generator.cleanup_after_image()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] question_id={question_id}，跳过")
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            print(f"[ERROR] question_id={question_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    generator.print_stats()

# =========================
# 主入口
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/root/autodl-tmp/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str,
                        default="/root/autodl-tmp/MME_Benchmark_release_version/MME_Benchmark")
    parser.add_argument("--question-file", type=str,
                        default="/root/autodl-tmp/Deco-main/eval_tool/eval/MME/llava_mme.jsonl")
    parser.add_argument("--answers-file", type=str,
                        default="/root/autodl-tmp/Deco-main/llava-answers_mme/ours_final/output_03.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    # parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1)

    # DeCo参数
    parser.add_argument("--use-deco", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)

    # MemVR参数
    parser.add_argument("--apply-memvr", action="store_true", default=True)
    parser.add_argument("--memvr-backward_layers", type=int, default=15)
    parser.add_argument("--memvr-retracing_ratio", type=float, default=0.25)
    parser.add_argument("--entropy_threshold_coef", type=float, default=0.75)

    # ED参数
    parser.add_argument("--use-ed", action="store_true", default=True)
    parser.add_argument("--ed-alpha", type=float, default=0.5)
    parser.add_argument("--ed-beta", type=float, default=0.5)
    parser.add_argument("--ed-tau", type=float, default=2.0)
    parser.add_argument("--ed-crop-size", type=int, default=336)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--max_samples", type=int, default=500)

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
