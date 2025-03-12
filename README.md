# SplitFed-Detection-Correction-Training

![License](https://img.shields.io/github/license/TIANGeng708/SplitFed-Detection-Correction-Training) ![Python](https://img.shields.io/badge/python-3.7%2B-blue)

## 🔥 项目简介
**SFL-DCT** 是首个面向 **分割联邦学习 (SplitFed Learning, SFL)** 的 **标签噪声鲁棒算法**，采用 **检测 (Detection)**、**纠正 (Correction)**、**训练 (Training)** 三阶段框架，旨在提升分割联邦学习在 **低质量数据** 场景下的稳定性和准确性。

### ✨ 主要特性
- **三阶段鲁棒训练框架**：检测、纠正、训练标签噪声数据，提高模型可靠性。
- **隐私保护**：无需访问原始数据，即可检测和修正标签噪声。
- **显著性能提升**：相比基准算法，SFL-DCT **最高提升 20.01%** 的准确率。
- **PyTorch 实现**，适用于不同数据集和模型架构。

## 🚀 快速开始
### 环境依赖
请确保安装以下依赖：
```bash
pip install torch torchvision numpy matplotlib
```

## ⚙️ 主要功能
| 功能 | 描述 |
|------|------|
| **标签噪声检测** | 识别数据中的错误标签 |
| **标签噪声纠正** | 通过优化策略修正标签错误 |
| **模型训练** | 在纠正后的数据上训练分割联邦学习模型 |
| **隐私保护** | 不访问原始数据即可完成噪声修正 |

## 📌 贡献指南
欢迎提交 **Issue** 和 **Pull Request** 来优化本项目！

## 📜 许可证
本项目遵循 **MIT License**，详情请见 [LICENSE](./LICENSE) 文件。

## 📎 相关链接
- 🔗 GitHub 仓库：[SFL-DCT](https://github.com/TIANGeng708/SplitFed-Detection-Correction-Training)

