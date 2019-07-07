
0. download_video_list.py
    Download ./yt_urls.json containing urls to videos.
    Download ./raw_data_cutup_jsons/<video_id>.json containing time stamp information.

1. ytvideos_2_mongodb.py:
	Download youtube video, convert to *.wav and save info to mongoDB
	save folder: "./raw_data"
	database: "simpsons"
	collection: "raw_data"

2. cut_and_label_2_mongodb.py:
	Read in *.wav signals of complete videos and create intervals for each character.
	The manually set base-intervals need to be defined in './raw_data_cutup_json/<video-id>.json
	After cuting and some pruning of the interval edges, resulting waves are saved to
	'./cut_and_labeled_data/<video-id> and infos stored in collection "cut_data".

3. create_snippets_simpsons.py
	Read in *.wav data of cut_and_labeled_data and create snippets of 2048 sample length. These
	snippets are only created on intervals with significant signal amplitude. These snippets
	are saved to ./snippets_2048/<character>/<uuid4> as *.wav and as *.npy. Infos are stored
	in "snippet_data" collection.

4. char_classifier.py
    yada yada