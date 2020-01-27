# vo-init

思想：双线程同时计算F和H模型，然后根据评分机制选择最优的模型

# 运行程序
需要提前下载tum数据集rgbd_dataset_freiburg1_desk：

    cd build
    cmake ..
    make -j4
    cd ../Examples/Monocular
    ./mono_tum TUM1.yaml ~/rgbd_dataset_freiburg1_desk
    
# 运行结果 

![result](https://github.com/liuzhenboo/vo-init/raw/master/result/1.png)
