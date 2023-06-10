# End-to-End ML Project for predicting house rental prices
Steps taken:
- [exploratory data analysis of dataset](https://github.com/ArtRakh/e2e23project/blob/main/EDA_real_estate_data.ipynb) from [Yandex.Realty](https://realty.ya.ru/) for the market of **St. Petersburg and Leningrad Oblast (2016-2018)**, the linear relationships investigation between cleaned factors provided:
![image](https://github.com/ArtRakh/e2e23project/assets/114469896/d2ec130b-0211-41f5-821d-e76e42c72179)
- [research](https://github.com/ArtRakh/e2e23project/blob/main/Building_model.ipynb) for a machine learning model based on [data](https://github.com/ArtRakh/e2e23project/blob/main/cleaned_dataset.csv) to solve the task (decision tree was chosen by MAPE), here is evidence for no overfitting:
  ```
  MAPE_valid = 21.84
  MAPE_test = 24.25
  ```
- using a remote machine (from Yandex.Cloud), running a ML method for predicting prices (both the model and the scalers were wrapped in [pickle files](https://github.com/ArtRakh/e2e23project/tree/main/mlmodels) which extracting is in the [file](https://github.com/ArtRakh/e2e23project/blob/main/Building_model.ipynb) with model building)
- using preloaded features in the function 'predict', execution of the GET method described in [app.py file](https://github.com/ArtRakh/e2e23project/blob/main/app.py) resulting a point prediction of house rental price
- building a [docker image](https://hub.docker.com/r/artrakh/e2e23_class_predictor) for the method to have an opportunity of running it within any environments (so without running app.py separately):
  ```
  FROM ubuntu:20.04
  MAINTAINER st110869
  RUN apt-get update -y
  COPY . /opt/gsom_predictor
  WORKDIR /opt/gsom_predictor
  RUN apt install -y python3-pip
  RUN pip3 install -r requirements.txt
  CMD python3 app.py
  ```
  In order not to run app.py file directly and help other potential users to predict prices in any environments, I packed pre-installed libraries (in requirements.txt), app.py, given the pickle files with model specified, and transformers into the docker image.
- running then the docker image from a primarily activated virtual environment in your virtual machine (commands in terminal):
  ```
  ssh <login of your virtual machine>@<public IP of the VM> # set a connection with your VM: my public IP: 51.250.20.152
  # dive into a folder with docker file like
  cd <folder name>
  source env/bin/activate # creating a self-contained directory tree that contains a Python installation for a particular version of Python
  docker run --network host -d <login of your virtual machine>/<folder with docker file within your machine>:v.0.2 # running the docker image, hence, our predicting algorithm, versions of the image may vary depending on your actions
  ```
- checking the correctness of the method by Postman (or another API platform) requesting a result by a public IP and a set of input parametres as seen in the screenshot below:
  ![image](https://github.com/ArtRakh/e2e23project/assets/114469896/b289d4d5-3a4f-4f12-acc1-57275de8944a)
  
  where:
  ```
  51.250.20.152 - public IP
  5444 - a port used in app.py via which you connect with a VM
  predict_price - function activating right after running app.py to get a point prediction with set of input parametres:
  open_plan=0&rooms=4&area=180&renovation=15
  ```
  #### In conclusion, there is a simple backend algorithm for implementing the house price prediction by means of virtual machine and docker images that improve reproducibility of the algorithm.
