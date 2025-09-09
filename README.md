<p align="center">
  <h1 align="center">Toward Zero-Shot Learning for Visual Dehazing of Urological Surgical Robots</h1>
  <p align="center">
    Renkai Wu, Xianjin Wang, Pengchen Liang, Zhenyu Zhang, Qing Chang* and Hao Tang*
  </p>
    <p align="center">
      1. Department of Surgery, Shanghai Key Laboratory of Gastric Neoplasms, Shanghai Institute of Digestive Surgery, Ruijin Hospital, Shanghai Jiao Tong University School of Medicine</br>
      2. National Key Laboratory for Multimedia Information Processing, School of Computer Science, Peking University</br>
      3. Department of Urology, Ruijin Hospital, Shanghai Jiaotong University School of Medicine</br>
      4. School of Intelligent Science and Technology, Nanjing University</br>
  </p>
</p>

## News🚀
(2025.03.31) ***The USRobot-Dehaze dataset and RSF-Dehaze code are publicly available***🔥🔥

(2025.01.28) ***The paper has been accepted by 2025 IEEE International Conference on Robotics & Automation -- ICRA 2025***🔥

(2024.10.02) ***The first edition of our paper has been uploaded to arXiv*** 📃

**0. Main Environments.**
- python 3.8
- pytorch 1.12.0

**1. Dataset acquisition and preparation** </br>

You can get the USRobot-Dehaze dataset from that [link](https://drive.google.com/file/d/1Tl8o5z4k1z18ahW4KX12xdYZDYBsEhgH/view?usp=sharing).

**2. Dehazing with real values** </br>

You can use the following command to dehaze test images in ./data:

```
python dehazing.py
```

**3. Dehazing without real values** </br>
If you want to test RSF-Dehaze on a image which does not have ground truth. You can use the following command:

```
python RW_dehazing.py
```

The only difference between two command is whether the program calculates PSNR and SSIM. We adhere to the baseline model--[YOLO](https://github.com/XLearning-SCU/2021-IJCV-YOLY), where the inference time is the rate of one iteration, and the number of iterations can be changed to improve efficiency.

## Citation

If you find RSF-Dehaze useful in your research, please consider citing:

```
@inproceedings{wu2025toward,
  title={Toward Zero-Shot Learning for Visual Dehazing of Urological Surgical Robots},
  author={Wu, Renkai and Wang, Xianjin and Liang, Pengchen and Zhang, Zhenyu and Chang, Qing and Tang, Hao},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4070--4076},
  year={2025},
  organization={IEEE}
}
```
## Acknowledgement
Thanks to [YOLO](https://github.com/XLearning-SCU/2021-IJCV-YOLY) for this outstanding work.
