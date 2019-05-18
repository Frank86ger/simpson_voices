
1. ytvideos_2_mongodb.py:
	Download youtube video, convert to *.wav and save info to mongoDB
	save folder: "./raw_data"
	database: "simpsons"
	collection: "raw_data"

2. cut_and_label_2_mongodb.py:
	Read in *.wav signal of complete video and create intervals for each character.
	The manually set base-intervals are defined in './raw_data_cutup_json/<video-id>.json
	After cuting and some pruning of the interval edges, resulting wave is saved to
	'./cut_and_labeled_data/<video-id> and infos stort in collection "cut_data".

3. asd.py
