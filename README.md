# DriverIdentifier

## Instructions
In case you too want to make your own classification model, for your own purposes, here is a list of instructions

1. Grab your images. Initially I had written a scraping script that grabs images off of google. However, I would recommend simply using a chrome extension to grab all images off of a page after you have searched for the person you want pictures of. This is also because Google is not a fan of people using web scraping on their services.

2. For each person, place their set of images in their own folder. Then place this folder inside the dataset folder. So for example, my two favourite drivers, Carlos Sainz and Yuki Tsunoda have their pictures in the following file paths: DriverIdentifier/model/dataset/CarlosSainz and DriverIdentifier/model/dataset/YukiTsunoda. Obviously, you replace the names with the person you are using. I strongly you suggest that you do not create a person called 'cropped' as this will cause some errors as a result of the data cleaning process.

3. Install the requirements that are specified in DriverIdentifier/model/requirements.txt. This can be done very easily with the following command: pip install -r requirements.txt

4. 