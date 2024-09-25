from langchain_text_splitters import RecursiveCharacterTextSplitter

text = '''SAFETY DATA SHEET 
5-15-2015 
 
 
 
 
Section 1: Identification 
Product Name: Marks -A-Lot Permanent Markers Inks 
 
Manufacturer Information 
Avery Products Corporation 
50 Pointe Drive 
Brea, California 92821 
Phone : 800 -462-8379 
 
Emergency Phone Number: 1-800-222-1222 (Poison Control Center) 
 
This product bears the AP (Approved Product -Non -Toxic) seal of The Art & Creative Materials Institute, Inc 
(ACMI). These products are certified by a medical expert to contain no materials in sufficient quantities to be toxic 
or injurious to humans or to cause acute or chronic health problems . 
 
Section 2: Hazard(s) Identification 
 
Classification in accordance with paragraph (d) of 29 CFR 1910.1200. 
Serious Eye Damage – Category 1 
Flammable Liquid 2 
 
GHS Label Elements 
Symbol(s) 
 
Signal Word (s) 
Danger 
 
Hazard Statement(s) 
Causes serious eye damage 
Highly flammable liquid vapor 
 
Precaution Statement(s) 
Prevention 
Keep away from heat, hot surfaces, sparks, open flames and other ignition sources. No smoking. Keep container 
tightly closed. Keep cool. Ground and bond container and receiving equipment. Use explosion -proof equipment.'''

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=150,
    length_function=len,
)

texts = text_splitter.split_text(text)
print(texts)
print(len(texts))