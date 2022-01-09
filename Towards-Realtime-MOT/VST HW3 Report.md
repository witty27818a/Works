# VST HW3 Report

---


# Experiment Setup


## Tracking Model



### Joint Detection and Embedding (JDE) model:


JDE is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network. 

Techical details are described in our ECCV 2020 paper. By using this repo, you can simply achieve MOTA 64%+ on the "private" protocol of MOT-16 challenge, and with a near real-time speed at 22~38 FPS (Note this speed is for the entire system, including the detection step! ) .

#### Model architecture



![](https://i.imgur.com/F1YcGEN.png)

![](https://i.imgur.com/9B8FtNR.png)


![](https://i.imgur.com/qb2j6ce.png)

從論文的這張table可以看出JDE的FPS相較當時其他SOTA的model進步非常多，非常適合這次作業使用webcam實時追蹤的task

---



# Code Explanation




## Ref : 
[Towards-Realtime_MOT](https://github.com/Zhongdao/Towards-Realtime-MOT)

## Usage : 
`python demo.py --input-video path/to/your/input/video --weights path/to/model/weights
               --output-format video --output-root path/to/output/root`
               
若從webcam接入則為
`python demo.py --input-video /dev/video0 --weights path/to/model/weights
               --output-format video --output-root path/to/output/root`
               
## Add mouse click feature

![](https://i.imgur.com/7vS6DDl.png)

single一開始initialize成False，每次點擊取not

![](https://i.imgur.com/9LKd7UA.png)
用single判斷是否開啟單物體追縱，並用trackid紀錄在追蹤的id

完整code link: https://github.com/franktseng0718/Towards-Realtime-MOT



---

# Demo video

從影片接入:
[demo1](https://www.youtube.com/watch?v=5ZM9gcHLjRs)

從webcam接入:
[demo1](https://www.youtube.com/watch?v=m64Fy4bUJgE)
[demo2](https://www.youtube.com/watch?v=OcCYeGwvISo)



---


# Discussion
這次作業看似簡單但我也搞了滿久，一開始方向有點想錯了，想說是不是要有兩個model交替使用才能達到滑鼠互動的效果，使用的那個github原本也沒有及時把每個frame show出來的功能，雖然只是要加一點點東西，但在加之前也必須看懂它整個codeflow，總共花的時間比預期多了滿多但做完後也滿有成就感的。

# Link 
[Hackmd Original file](https://hackmd.io/fXWf7Hg4RDe2nAMwbcASqA?both)





