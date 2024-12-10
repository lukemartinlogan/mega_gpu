# Dependencies:
* cuda
* hermes_shm

## Spack
```bash
git clone https://github.com/spack/spack.git
cd spack
git checkout tags/v0.22.2
echo ". ${PWD}/share/spack/setup-env.sh" >> ~/.bashrc
source ~/.bashrc
```

## hermes-shm
```
git clone https://github.com/lukemartinlogan/grc-repo.git
cd grc-repo
spack repo add .
spack install hermes_shm@master
```

# Compiling
```bash
spack load hermes_shm
mkdir build
cd build
cmake ../
make -j32
```

# Running
```bash
build/test_basic
```
