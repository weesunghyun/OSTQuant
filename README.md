# AAQ: Alignment-Aware Quantization for LLM Safety

Official code for the paper:

**Alignment-Aware Quantization for LLM Safety**  
Sunghyun Wee, Juhwan Cho, Seoyeon Park, Nojun Kwak†  
Department of Electrical and Computer Engineering, Seoul National University

[[Paper (Coming Soon)]()]  
[[Project Page (TBA)]()]  
[[License](./LICENSE)]

---

## 🧠 Overview

Post-training quantization (PTQ) is widely used to compress large language models (LLMs), but existing methods often ignore safety alignment from RLHF or instruction tuning. As a result, safety behavior may collapse even when perplexity remains stable.

We propose **Alignment-Aware Quantization (AAQ)**, a novel PTQ framework that integrates a contrastive KL loss to explicitly preserve alignment during quantization. Our method:
- Steers the quantized model towards the aligned model’s output
- Pushes it away from the pre-trained base model
- Is lightweight and data-efficient (only needs calibration data)

Tested on the LLaMA family (LLaMA2-Chat, LLaMA3-Instruct, etc.), AAQ achieves high safety performance under 4-bit quantization.

---

## 📦 Installation

```bash
git clone https://github.com/weesunghyun/AAQ.git
cd AAQ
conda create -n aaq python=3.10
conda activate aaq
pip install -r requirements.txt
```


## Citation
If you find OSTQuant useful in your research, please consider citing our paper:
```bibtex
@misc{wee2025aaq,
  title={Alignment-Aware Quantization for LLM Safety},
  author={Sunghyun Wee and Juhwan Cho and Seoyeon Park and Nojun Kwak},
  year={2025},
  note={arXiv:xxxx.xxxxx},
  url={https://github.com/weesunghyun/AAQ}
}
```
