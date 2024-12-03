# The problem

We need to build a system that can:
1) Identify unique influencers appearing across these videos
2) Calculate the average performance for each influencer 
3) Help understand which influencers consistently drive better engagement

# Solution v0

1) We start by fetching the videos from the CSV file and store them in `/data`.
2) Initially, face detection is performed using OpenCV and they are then filtered based on image dimensions.
3) Similar faces are first clustered using DBSCAN and then agglomerative clustering to identify unique influencers.
4) Optionally, a second pass can be performed over the unclustered images for face detection using RetinaFace.
4) Once this is done, we build an inverted index of faces i.e. map each influencer to the set of videos they appear in. Videos with no faces are ignored.
5) There are some duplicate videos in there so we consider the version with maximum performance as it is, and multiply the other scores by some damping factor between 0 and 1.
6) For each influencer, we calculate the mean, median, variance and standard deviation of their performance.
7) We then assign each influencer a 'consistency score' based on these values, which ranges from 1-5 (1 = highly inconsistent, 5 = highly consistent) so that it can be easily interpreted by non-technical people.
8) Finally, we display the face of each unique influencer along with their corresponding performance and consistency scores.
9) A PDF/HTML report containing this table is automatically generated.

# Solution v1

# Challenges

1) Dealing with false positives
2) Segregating real influencer faces from photographic faces
3) Clustering faces (DBSCAN created multiple clusters for the same person)

# Brownie points

1) End-to-end pipeline with simple one click GUI
2) 2-pass filter
3) Ranking each influencer
4) Listing their attributes (gender, age, similarity, etc.)