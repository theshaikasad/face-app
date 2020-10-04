import streamlit as st 
import os
import numpy as np
from PIL import Image
import numpy as np
from deepface import DeepFace
from deepface.basemodels import Facenet
from deepface.commons import functions, distance as dst

import pickle

model = Facenet.loadModel()

with open('Actors/representations_facenet.pkl', 'rb') as f:
	actorso = pickle.load(f)

with open('Actress/representations_facenet.pkl', 'rb') as f:
	actress = pickle.load(f)


def main(): 

	""" Our App """

	st.title("Find what indian actor/actress you look like") 
	st.text("Built with streamlit and deepface, please wait for 20 seconds for the result")
	st.set_option('deprecation.showfileUploaderEncoding', False)
	image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])


	if image_file is not None:
		st.write("Upload Successful")

		image = Image.open(image_file)

		open_cv_test_image = np.array(image) 
		open_cv_test_image = open_cv_test_image[:, :, ::-1].copy() # Convert RGB to BGR

		input_shape = model.layers[0].input_shape
					
		if type(input_shape) == list:
			input_shape = input_shape[0][1:3]
		else:
			input_shape = input_shape[1:3]

		demography = DeepFace.analyze(open_cv_test_image, actions=["gender"])
		
		if demography["gender"] == "Man":
			entries = actorso
		else:
			entries = actress

		test_preds = model.predict(functions.preprocess_face(img=open_cv_test_image, target_size = input_shape, enforce_detection=False, detector_backend='ssd'))[0, :]
		
		least_diff = dst.findEuclideanDistance(dst.l2_normalize(entries[0][1]), dst.l2_normalize(test_preds))
		least_img = entries[0][0]
		print(least_img)

		for rep in entries:
			diff = dst.findEuclideanDistance(dst.l2_normalize(rep[1]), dst.l2_normalize(test_preds))
			print(rep[0])
			if(diff<least_diff):
				least_diff = diff
				least_img = rep[0]

		st.write("You look like ",  least_img[8:][:-4]) 
		pil_result_im = Image.open(least_img)
		st.image(pil_result_im, width=140)


if __name__ == "__main__":
	main()
   
