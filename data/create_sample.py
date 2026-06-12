"""
Run this script once to generate sample_reviews.csv for demo purposes.
Usage: py data/create_sample.py
"""
import pandas as pd
import os

reviews = [
    {"review_text": "Amazing product! Works perfectly with my iPhone. Found my keys instantly.", "rating": 5},
    {"review_text": "Battery life is terrible. Dies in less than 2 days. Very disappointed.", "rating": 1},
    {"review_text": "Good tracker but a bit expensive for what it does.", "rating": 3},
    {"review_text": "Lost my keys and found them immediately. Love this product.", "rating": 5},
    {"review_text": "Not waterproof at all. Stopped working after getting caught in the rain.", "rating": 2},
    {"review_text": "Works great but the app needs improvement. Initial setup was confusing.", "rating": 3},
    {"review_text": "Perfect for tracking luggage during international travel. Highly recommend!", "rating": 5},
    {"review_text": "Received a defective unit. Does not connect to Bluetooth at all.", "rating": 1},
    {"review_text": "Decent product. Does what it says but nothing extraordinary.", "rating": 3},
    {"review_text": "My dog keeps chewing it off his collar. Not durable enough for pets.", "rating": 2},
    {"review_text": "Best investment I made this year. Never lose my wallet again!", "rating": 5},
    {"review_text": "The precision finding feature is incredible. Very accurate indoors.", "rating": 5},
    {"review_text": "Privacy concerns. Anyone with an iPhone can track my location unknowingly.", "rating": 2},
    {"review_text": "Stopped working after 3 months. Poor build quality for the price.", "rating": 1},
    {"review_text": "Great for finding things but only works well within the Apple ecosystem.", "rating": 4},
    {"review_text": "Super easy to set up. My kids love using it to find their backpacks.", "rating": 5},
    {"review_text": "The anti-stalker alert is appreciated but it still feels somewhat invasive.", "rating": 3},
    {"review_text": "Bought 4 of them for the whole family. Worth every dollar spent.", "rating": 5},
    {"review_text": "Sound is too loud for indoor use. Woke my baby up when it started beeping.", "rating": 2},
    {"review_text": "Fast and reliable. Saved me from missing my flight by finding my bag.", "rating": 5},
    {"review_text": "Not bad but competitors offer similar features at a lower price point.", "rating": 3},
    {"review_text": "The app crashes every time I try to use the precision finding mode.", "rating": 1},
    {"review_text": "Excellent product! The UWB chip makes it far more accurate than alternatives.", "rating": 5},
    {"review_text": "Too dependent on the Apple network. Poor performance in rural areas.", "rating": 2},
    {"review_text": "Setup took 2 minutes. Works seamlessly across all my Apple devices.", "rating": 5},
    {"review_text": "Not compatible with Android. Very misleading product description on the box.", "rating": 1},
    {"review_text": "Decent battery life but the replacement CR2032 battery costs more than expected.", "rating": 3},
    {"review_text": "Used it to track my bike. Found it 2 blocks away after it was stolen!", "rating": 5},
    {"review_text": "The precision finding vibration feedback is a nice touch but drains battery fast.", "rating": 4},
    {"review_text": "Poor customer service. Took 3 weeks to get a replacement for defective unit.", "rating": 1},
    {"review_text": "Compact design fits anywhere. No complaints after 6 months of daily use.", "rating": 5},
    {"review_text": "Keeps disconnecting from my iPhone 13. Software bugs need to be fixed urgently.", "rating": 2},
    {"review_text": "Very helpful for forgetful people like me. Attached to everything I own.", "rating": 5},
    {"review_text": "Packaging was damaged on arrival. Product itself seems fine after testing.", "rating": 3},
    {"review_text": "Impressive range. Still connected from 50 metres away through multiple walls.", "rating": 5},
]

output_path = os.path.join(os.path.dirname(__file__), 'sample_reviews.csv')
df = pd.DataFrame(reviews)
df.to_csv(output_path, index=False)
print(f"✅ Created {len(df)} sample reviews at: {output_path}")
