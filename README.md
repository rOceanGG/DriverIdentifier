# DriverIdentifier

![](https://github.com/rOceanGG/DriverIdentifier/blob/main/DriverIdentifierDemo.gif)

## Wanna run it on your machine?
1. Clone this repository.
2. Install all required modules.
3. Initiate the flask server by running DriverIdentifier/server/server.py (Ensure that you do not already have port 5000 being used).
4. Use your choice of web browser and paste the complete file path to DriverIdentifier/UI/app.html
5. Upload your choice of image and try it out for yourself.

## Instructions for your own model
In case you too want to make your own classification model, for your own purposes, here is a list of instructions.

1. Grab your images. Initially I had written a scraping script that grabs images off of google. However, I would recommend simply using a chrome extension to grab all images off of a page after you have searched for the person you want pictures of. This is also because Google is not a fan of people using web scraping on their services.

2. For each person, place their set of images in their own folder. Then place this folder inside the dataset folder. So for example, my two favourite drivers, Carlos Sainz and Yuki Tsunoda have their pictures in the following file paths: DriverIdentifier/model/dataset/CarlosSainz and DriverIdentifier/model/dataset/YukiTsunoda. Obviously, you replace the names with the person you are using. I strongly you suggest that you do not create a person called 'cropped' as this will cause some errors as a result of the data cleaning process.

3. Install the requirements that are specified in DriverIdentifier/model/requirements.txt. This can be done very easily with the following command: pip install -r requirements.txt.

4. Set a break point at the line saying 'self.createClassificationDictionary()'. Run a python debugger. This will pause the program just after the getFaces() method is called.

5. Go into your cropped directory in your dataset and delete any images that are either not faces, or faces of people you are not looking for.

6. Let the program run until it terminates.

7. You should now have a file called savedModel.pkl. This is your classification model. Do with it as you please :).
