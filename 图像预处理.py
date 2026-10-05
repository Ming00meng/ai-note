# 依赖安装：
#   pip install requests opencv-python numpy
import json
import os
import sys
import time
import base64

import cv2
import numpy as np
import requests

# ============================================================
# 配置：TOKEN 建议用环境变量，不要把真实 token 写死进代码
#   export PADDLE_OCR_TOKEN="你的token"
# 注意：你刚才贴的代码里的 token 已经公开暴露了，强烈建议去
#       AI Studio 重新生成一个，并在控制台把旧的吊销掉。
# ============================================================
JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
TOKEN   = os.getenv("PADDLE_OCR_TOKEN", "0e7f00b02fbf610544e842f884431d7a87685ed8")
MODEL   = "PP-OCRv6"

# 预处理开关：True=先用 OpenCV 去噪+二值化再上传；图本身较干净可设 False
ENABLE_PREPROCESS = True
ROI_CENTER        = False   # True=只送图像中间区域，False=送整图


# ------------------------------------------------------------
# 1. OpenCV 预处理：灰度 + 高斯/中值去噪 + Otsu 二值化
# ------------------------------------------------------------
def preprocess(image_path: str, save_as: str = "_preprocessed.png") -> str:
    src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if src is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    denoised = cv2.medianBlur(denoised, 3)

    # 中值滤波可能让笔画变糊，若原图已较干净，可注释掉 medianBlur
    _, binary = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = binary
    if ROI_CENTER:
        h, w = img.shape
        img = img[h // 4: h * 3 // 4, w // 4: w * 3 // 4]

    out = save_as if os.path.isabs(save_as) else os.path.join(
        os.path.dirname(os.path.abspath(image_path)), save_as)
    cv2.imwrite(out, img)
    print(f"预处理完成，已保存: {out}")
    return out


# ------------------------------------------------------------
# 2. 提交异步任务（官方 Jobs 接口）
# ------------------------------------------------------------
def submit_job(file_path: str) -> str:
    headers = {"Authorization": f"bearer {TOKEN}"}
    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }

    if file_path.startswith("http"):
        headers["Content-Type"] = "application/json"
        payload = {"fileUrl": file_path, "model": MODEL,
                   "optionalPayload": optional_payload}
        r = requests.post(JOB_URL, json=payload, headers=headers)
    else:
        if not os.path.exists(file_path):
            sys.exit(f"Error: File not found at {file_path}")
        data = {"model": MODEL,
                "optionalPayload": json.dumps(optional_payload)}
        with open(file_path, "rb") as f:
            r = requests.post(JOB_URL, headers=headers,
                              data=data, files={"file": f})

    print(f"Response status: {r.status_code}")
    if r.status_code != 200:
        print(f"Response content: {r.text}")
    r.raise_for_status()
    job_id = r.json()["data"]["jobId"]
    print(f"Job submitted successfully. job id: {job_id}")
    return job_id


# ------------------------------------------------------------
# 3. 轮询直到 done / failed
# ------------------------------------------------------------
def poll_job(job_id: str) -> str:
    headers = {"Authorization": f"bearer {TOKEN}"}
    while True:
        r = requests.get(f"{JOB_URL}/{job_id}", headers=headers)
        r.raise_for_status()
        data = r.json()["data"]
        state = data["state"]

        if state == "pending":
            print("job 状态: pending")
        elif state == "running":
            prog = data.get("extractProgress", {})
            print(f"job 状态: running, "
                  f"总页 {prog.get('totalPages','?')} / "
                  f"已处理 {prog.get('extractedPages','?')}")
        elif state == "done":
            prog = data["extractProgress"]
            print(f"Job 完成, 抽取页数: {prog['extractedPages']}, "
                  f"开始: {prog['startTime']}, 结束: {prog['endTime']}")
            return data["resultUrl"]["jsonUrl"]
        elif state == "failed":
            sys.exit(f"Job failed, 原因: {data.get('errorMsg')}")

        time.sleep（5）


# ------------------------------------------------------------
# 4. 解析 jsonl：核心是取出每页的 rec_texts（识别文本）
#    官方示例只存了标注图，这里补上文字落盘
# ------------------------------------------------------------
def parse_result(jsonl_url: str, out_txt: str = "output.txt",
                 save_images: bool = True):
    lines = requests.get(jsonl_url).text.strip().split("\n")
    os.makedirs("output", exist_ok=True)

    all_text = []
    page_num = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        result = json.loads(line)["result"]
        for res in result.get("ocrResults", []):
            pruned = res.get("prunedResult", {})
            texts = pruned.get("rec_texts", [])
            for t in texts:
                all_text.append(t)

            # 可选：下载带框标注图
            if save_images:
                img_url = res.get("ocrImage")
                if img_url:
                    ir = requests.get(img_url)
                    if ir.status_code == 200:
                        fn = f"outputimg_output_{page_num}.jpg"
                        with open(fn, "wb") as f:
                            f.write(ir.content)
                        print(f"标注图已保存: {fn}")
            page_num += 1

    text = "\n".join(all_text)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n===== OCR 结果（共 {len(all_text)} 行）=====")
    print(text)
    print(f"\n结果已保存到: {out_txt}")


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "input.jpg"
    out_txt   = sys.argv[2] if len(sys.argv) > 2 else "output.txt"

    target = file_path
    if ENABLE_PREPROCESS and not file_path.startswith("http"):
        target = preprocess(file_path)

    print(f"Processing file: {target}")
    job_id = submit_job(target)
    jsonl_url = poll_job(job_id)
    parse_result(jsonl_url, out_txt)


if __name__ == "__main__":
    main()