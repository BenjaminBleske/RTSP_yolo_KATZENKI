Installation notwendige Pakete:


sudo apt install python3-pip
pip3 install flask opencv-python
sudo apt install libgl1

mkdir ~/rtsp_webapp
cd ~/rtsp_webapp

python3 -m venv venv
source venv/bin/activate

pip install flask opencv-python

nohup python rtsp_inference.py &


++++++
um das zu starten ohne ssh damit es weiterläuft

nohup ./venv/bin/python rtsp_inference.py > rtsp.log 2>&1 &



damit es beendet wird

ps aux | grep rtsp_inference.py
pkill -f rtsp_inference.py
