# TIM Messenger - AI assistant for medical history assessment

## Install requirements (Ubuntu)
Check you CUDA Version
> nvidia-smi

Install cublas and cuda compiler
> apt-get install libcublas-dev-12-4 (should match your CUDA Version)
> apt-get install cuda-nvcc-12-4

## Install tim.assistant
Clone Repo
> git clone https://github.com/nonelabs/tim.assistant
> cd tim.assistant
> ./download_models.sh
> ./install_venv.sh
> source venv/bin/activate

## Configuration
> copy env_template .env
> 
## Modify a .env file in your project root
> TIM_USERNAME="USER"
> TIM_PASSWORD="PASSWORD"
> TIM_HOMESERVER="matrix.org"

## Run
> python src/main -q query_groups
