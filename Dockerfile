from python:3.10.11

run echo "export PATH=\"/root/.local/bin:$PATH\"" >> /root/.bashrc
run echo "alias ls='ls --color=auto'" >> /root/.bashrc
run echo "alias ll='ls -alF'" >> /root/.bashrc
run echo "alias la='ls -A'" >> /root/.bashrc
run echo "alias l='ls -CF'" >> /root/.bashrc

run sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list
run apt update && apt install -y vim curl wget

workdir /workspace

run pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple langchain-text-splitters==0.2.2
add requirements.txt requirements.txt
run pip install -r requirements.txt
run pip uninstall -y graphrag

add graphrag graphrag
add template template
add template_zh template_zh
add index.py index.py
add serving.py serving.py
add init.py init.py
