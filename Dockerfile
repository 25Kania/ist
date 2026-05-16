FROM apache/spark:3.5.4-scala2.12-java11-python3-ubuntu

USER root

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run a bash shell
CMD ["bash"]

# ===== BUILD COMMANDS =====
# To build this Docker image, run:
# docker build -t bt-app .
#
# ===== RUN COMMANDS =====
# To run the container in interactive mode with current working directory as volume:
# 
# On Linux/Mac:
# docker run -it --rm -v $(pwd):/app bt-app:latest
#
# On Windows (PowerShell):
# docker run -it --rm -v ${PWD}:/app bt-app:latest
#
# On Windows (Command Prompt):
# docker run -it --rm -v %CD%:/app bt-app:latest
#
# To run with a bash shell for development:
# docker run -it --rm -v $(pwd):/app bt-app:latest bash
# ===== END COMMANDS =====