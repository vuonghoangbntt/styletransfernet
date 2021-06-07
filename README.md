# styletransfernet
This is a simple style transfer app made with Streamlit and pytorch implementation of a paper Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017].
Model is trained by our team and saved in file `check_point_epoch_35_10000_samples_v2.pth`
## Requirement
Install all the requirements by : `pip install -r requirements.txt`
- Python 3.8+
- PyTorch 1.8+
- TorchVision  0.9+
- Streamlit 0.62+

## Usage
### Run app local
Firstly, download all requirement and run in powershell `../styletransfernet/streamlit run app.py` then the app will show in your default browser
### Deploy in website
We deployed this app with streamlit share host in this link `https://share.streamlit.io/vuonghoangbntt/styletransfernet/main/app.py`. Try it without downloading to your local computer
## References
- [1]: X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.


