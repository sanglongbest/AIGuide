# Caffe在aws上的安装流程

标签（空格分隔）： 人工智能 Caffe

---

#前言
本文记录了在新安装的 ubuntu 14.04 系统下安装 caffe 的过程。这里主要参考：
http://coldmooon.github.io/2015/08/03/caffe_install/

#安装 nvidia 显卡驱动

如果电脑没有 nvidia 的显卡，此步跳过。

网上的许多教程都指出要进入 tty，然后把 lightdm 关了。但我发现直接用 apt-get 安装的话，无需关闭 lightdm。

也有很多网络教程采用 apt-get 来进行显卡安装，这个在 AWS 上是不可取，容易产生 CUDA 的错误。

所以请用

```
lspci |grep VGA
```
来查询显卡型号，然后上 NVIDA 官网下载响应的驱动。

安装完后，输入 `prime-select query` 查看当前正在使用的显卡。
```
$ prime-select query
nvidia
```
输入 `cat /proc/driver/nvidia/version` 查看正在使用的 nvidia 驱动版本和编译时采用的 gcc 版本
```
$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  352.30  Tue Jul 21 18:53:45 PDT 2015
GCC version:  gcc version 4.9.2 (Ubuntu 4.9.2-0ubuntu1~14.04)
```
虽然 cuda 里已经包含了 nvidia 驱动，但是根据 caffe 官方指导，cuda 与显卡驱动最好分开安装。

参考链接:
http://ubuntuhandbook.org/index.php/2015/04/install-nvidia-driver-346-59-in-ubuntu-from-ppa/
http://www.binarytides.com/install-nvidia-drivers-ubuntu-14-04/
http://my.oschina.net/eechen/blog/227134
https://devtalk.nvidia.com/default/topic/810964/linux/black-screen-after-prime-select-nvidia-and-log-out-using-v346-35-drivers/2/

#安装 cuda 7.0
如果电脑没有 nvidia 的显卡，此步跳过。有人说即使电脑上没有 nvidia 显卡也必须装 cuda, 否则会出问题。但我亲自实验过，完全可以不装 cuda。但必须在 makefile 文件中把 `CPU_ONLY := 1` 打开。并且不能使用 cuda 相关函数。如果使用 cuda 相关函数，则会报错。

从 cuda 官方网站 下载对应的 deb包。然后双击，在软件中心里安装。网站提供两个下载版本一个完全版，一个是索引版。我选的后者，这样上传到 AWS 方便。

然后就按照网站的提示进行安装。
接下来输入下列命令安装 cuda:
```
sudo apt-get update
sudo apt-get install cuda
```
安装完成后，再配置环境
```
export PATH=/usr/local/cuda-7.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
```

#安装 cudnn
如果电脑没有 nvidia 的显卡，此步跳过。从官方网站下载 cudnn 后解压。得到的文件是 .h 和 .so 文件。所以，直接把他们拷贝到 /usr/local/include 和 /usr/local/lib/ 下就好了。
```
sudo cp cudnn.h /usr/local/include
sudo cp libcudnn.so.X.Y.ZZ /usr/local/lib //so.X.Y.ZZ自行对应相应版本

sudo ln -s /usr/local/lib/libcudnn.so.X.Y.ZZ /usr/local/lib/libcudnn.so.X.Y
sudo ln -s /usr/local/lib/libcudnn.so.X.Y /usr/local/lib/libcudnn.so

sudo ldconfig
```
注意: 检查一下刚刚拷贝到 /usr/local/lib 下的 libcudnn.so.X.Y.ZZ 的文件权限。
```
$ ls -l *cudnn*

lrwxrwxrwx 1 root root       33  8月  4 22:05 libcudnn.so -> /usr/local/lib/libcudnn.so.6.5.48
lrwxrwxrwx 1 root root       18  8月  4 22:09 libcudnn.so.6.5 -> libcudnn.so.6.5.48
-rw------- 1 root root 11172416  8月  2 23:18 libcudnn.so.6.5.48
-rw------- 1 root root 11623922  8月  2 23:19 libcudnn_static.a
```
从上面的显示结果可以看到，so.X.Y.ZZ 对于 others 用户是没有读取权限的，这会导致编译 caffe时出现下列错误:
```
AR -o .build_release/lib/libcaffe.a
LD -o .build_release/lib/libcaffe.so
/usr/bin/ld: cannot find -lcudnn
collect2: error: ld returned 1 exit status
make: *** [.build_release/lib/libcaffe.so] Error 1
```
解决方法很简单，只要赋予 others 可读(写)权限即可:
```
sudo chmod 755 libcudnn.so.X.Y.ZZ
```

#安装 anaconda
强烈推荐使用 anaconda 的python。它里面集成了很多包，ipython, mkl, numpy等都预装了 省去了很多麻烦。如果有 edu 邮箱的话，还可以获得 anaconda accelerate，在矩阵运算的时候，可以启用并行计算，速度快很多。

这也是要先上官网下载，推荐用浏览器查看下载地址，然后在 aws上用`wget`下载。
```
安装 anaconda:
./Anaconda-2.3.0-Linux-x86_64.sh

安装 accelerate:
conda update conda
conda install accelerate
conda install iopro
```
接下来拷贝 anaconda 的许可文件到用户主目录
```
mv license_academic_20150611072013.txt ~/.continuum
```

然后升级 ipython, 如果不用 ipython，那就跳过下面这一步:
```
conda update ipython
conda update ipython-notebook
conda update ipython-qtconsole
```
下面测试一下 anaconda python 的功能，首先在终端下启用
```
ipython-notebook

$ ipython notebook
```
然后新建一个 ipynb 文件。在 cell 中输入
```
In [1]:  import mkl
         mkl.set_num_threads(4) # 设置最大线程数
         mkl.get_max_threads()  # 查看当前线程数

Out[1]:  4
```
可见 anaconda 已经预装了 MKL。测一下速度:
```
In [23]: a = np.random.random((4096, 4096))
         %timeit np.dot(a,a)

         1 loops, best of 3: 6.44 s per loop
```
# 安装 Opencv
因为喜欢opencv 3.0 的，所以安装了 opencv 3.0。然后就掉入了坑，因为 caffe 对 opencv 3.0支持那是相当不好。

Opencv 的安装过程较繁琐，且网上已经推荐三个:
http://www.sysads.co.uk/2014/05/install-opencv-2-4-9-ubuntu-14-04-13-10/
http://rodrigoberriel.com/2014/10/installing-opencv-3-0-0-on-ubuntu-14-04/
http://karytech.blogspot.hu/2012/05/opencv-24-on-ubuntu-1204.html

```
这里提供一个脚本，方便在 terminal 下编译 Opencv 程序。
脚本来自: https://jayrambhia.wordpress.com/2012/05/08/beginning-opencv/

echo "compiling $1"
if [[ $1 == *.c ]]
then
    gcc -g `pkg-config --cflags opencv`  -o `dirname $1`/`basename $1 .c` $1 `pkg-config --libs opencv`;
elif [[ $1 == *.cpp ]]
then
    g++ -g `pkg-config --cflags opencv` -std=c++11 -std=gnu++11 -o `dirname $1`/`basename $1 .cpp` $1 `pkg-config --libs opencv`;
else
    echo "Please compile only .c or .cpp files"
fi
echo "Output file => ${1%.*}"
```

将上述代码保存为一个 xxx.sh 文件，名字自己起。然后在终端里给该文件开启可执行权限:
```
sudo chmod 777 xxx.sh
```
接下来在 .bashrc 中建立一个 alias 来指向 xxx.sh
```
subl ~/.bashrc
```
在 .bashrc 中键入
```
alias opencv="/path/to/xxx.sh"
```
以后要编译 opencv 程序的时候，只需要在终端里输入 opencv xxx.cpp 即可。无需敲入繁琐的 pkg-config 前后缀。例如, 直接在终端里键入 opencv 命令，会提示
```
$ opencv

compiling
Please compile only .c or .cpp files
Output file =>
```
其他可选教程：
https://nusharex.wordpress.com/2015/06/01/18/
快捷安装脚本：
https://gist.github.com/Coldmooon/c2e146bb7e960556e055

如果你用的是 OpenCV3.0，请打开Makefile
在文件最后添加：
```
 LIBRARIES += glog gflags protobuf leveldb snappy \
  lmdb boost_system hdf5_hl hdf5 m \
  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
```
不然会报错，因为 highgui 换位置了。

#安装 MKL, Openblas or Atlas
##MKL
其实 anaconda 已经自带了 MKL, 但不不妨这里再装一下。首先去下面的链接下载学生版 MKL
https://software.intel.com/en-us/intel-mkl
安装过程不说了，基本直接下一步就可以了。接下来配置环境:
```
sudo vim /etc/ld.so.conf.d/intel_mkl.conf
```
输入:
```
/opt/intel/lib/intel64
/opt/intel/mkl/lib/intel64
```
关闭并保存文件。
```
sudo ldconfig
– Openblas
```
去官方网站 下载 OpenBLAS 的安装包，解压。进入安装目录，在终端输入
```
$ make -j 8
```
安装成功之后，继续在终端输入
```
$ make install PREFIX=your_directory
```
注意，如果这里自己选择安装目录，则在编译 Caffe 的时候，会提示找不到 OpenBLAS的库文件，此时，需要进一步设置 LD_LIBRARY_PATH 才行。

根据具体的安装路径设置:
```
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib
```
##Atlas
```
sudo apt-get install libatlas-base-dev
```
#安装 Boost
进入 Boost 的官方网站 http://www.boost.org/ 下载安装包。按照官方指南进行安装。
最简单的方法是直接在 ubuntu 的软件仓库里搜索 libboost。也在可以用 apt-get 安装:
```
sudo apt-get install libboost-all-dev
```
其实这一步可以放到下一节来做。

##其他依赖库
按照 官方指南 进行:
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```
安装过 anaconda 的话，那 libhdf5-serial-dev 可以不装。如果编译时提示找不到 hdf5 的库。就把 anaconda/lib 加到 ld.so.conf 中去。
```
$ sudo vim /etc/ld.so.conf
```
添加一行,用户名改为你自己的:
```
/home/your_username/anaconda/lib
```
关闭并保存文件。
```
$ sudo ldconfig
```
#编辑 Caffe makefile 文件:
重点要改的地方，电脑有 nvidia 显卡的配置:
```
# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
# CPU_ONLY := 1

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
                -gencode arch=compute_20,code=sm_21 \
                -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode arch=compute_50,code=sm_50 \
                -gencode arch=compute_50,code=compute_50

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := mkl
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# BLAS_INCLUDE := /path/to/your/blas
# BLAS_LIB := /path/to/your/blas

# MATLAB directory should contain the mex binary in /bin.
MATLAB_DIR := /home/your_username/MATLAB/R2014b
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# PYTHON_INCLUDE := /usr/include/python2.7 \
#               /usr/lib/python2.7/dist-packages/numpy/core/include
# Anaconda Python distribution is quite popular. Include path:
# Verify anaconda location, sometimes it's in root.
ANACONDA_HOME := $(HOME)/anaconda
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
                $(ANACONDA_HOME)/include/python2.7 \
                $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \

# We need to be able to find libpythonX.X.so or .dylib.
# PYTHON_LIB := /usr/lib
PYTHON_LIB := $(ANACONDA_HOME)/lib

# Uncomment to support layers written in Python (will link against Python libs)
# WITH_PYTHON_LAYER := 1
```
电脑没有 nvidia 显卡的配置:
```
# cuDNN acceleration switch (uncomment to build with cuDNN).
# USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
CPU_ONLY := 1

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
# CUDA_DIR := /usr/local/cuda
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them.
# For CUDA < 6.0, comment the *_50 lines for compatibility.
# CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
#                 -gencode arch=compute_20,code=sm_21 \
#                 -gencode arch=compute_30,code=sm_30 \
#                 -gencode arch=compute_35,code=sm_35 \
#                 -gencode arch=compute_50,code=sm_50 \
#                 -gencode arch=compute_50,code=compute_50

...
```
其余部分保持不变
接下来编译。

#编译
在终端下输入:
```
$ mkdir build && cd build
$ cmake ..
$ make all -j 8
$ make install
$ make test -j 8
$ make runtest -j 8
```
接下来编译 python 接口和 matlab 接口:
```
$ make pycaffe -j 8
$ make matcaffe -j 8
```
