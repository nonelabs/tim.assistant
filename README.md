# TIM Messenger - AI assistant for medical history assessment

Example Python implementation of an LLM-based assistant that can conduct medical history interviews through the Matrix messaging platform. 
The focus is on simplicity and comprehensibility rather than employing state-of-the-art methods. 

## Install requirements (Ubuntu)
Install NVIDIA drivers and CUDA if not yet done.

Figure out CUDA Version
```
nvidia-smi
```
![image](https://github.com/user-attachments/assets/07d4a27a-da4f-473d-82b2-e69b0e86fb41)

Install cublas and cuda compiler
```
apt-get install libcublas-dev-12-4 (should match your CUDA Version)
apt-get install cuda-nvcc-12-4
```
## Install tim.assistant
Clone Repo
```
git clone https://github.com/nonelabs/tim.assistant
cd tim.assistant
./download_models.sh
./install_venv.sh
source venv/bin/activate
```
## Configuration
```
copy env_template .env
```
## Define homeserver and credentials

Edit .env file

```
TIM_USERNAME="USER"
TIM_PASSWORD="PASSWORD"
TIM_HOMESERVER="matrix.org"
```
## Run
```
python src/main -q query_groups
```
