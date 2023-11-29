# Use a base image with Python 3.9 (an AWS Lambda-compatible version)
FROM public.ecr.aws/lambda/python:3.8

# Copy requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Adjusted the path to copy the 'src' directory

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

COPY ./* ./inpaint-t/

COPY migan_512_places2.pt .
COPY lib ./lib
COPY dnnlib ./dnnlib
COPY torch_utils ./torch_utils
COPY src ./src  
# Install dependencies (you may want to use Python 3.9)

# Set the Lambda function handler
CMD [ "app.handler" ]

#COPY app.py ${LAMBDA_TASK_ROOT}

# Install the specified packages


# Set the CMD to your h