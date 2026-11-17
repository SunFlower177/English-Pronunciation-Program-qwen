# MultiPA: A Multi-task Speech Pronunciation Assessment Model for Open Response Scenarios


MultiPA data
The multiPA dataset comprises 50 audio clips, each lasting 10 to 20 seconds, obtained from anonymous dialog chatbot users. These users were native Mandarin speakers who used the dialog chatbot to practice their English. As a real use case for automatic pronunciation assessment, users can access the system with their own headsets at their own location. 

Five annotators made the scores, each annotator scores independently under the same metric.

**Sentence level**

Accuracy (1-5)
* 5 (Excellent. The overall pronunciation of the sentence is nearly perfect with accurate articulation of all sounds.)
* 4 (Good. Minor pronunciation errors may be present, but overall, the speaker's pronunciation is highly accurate and easily understandable.)
* 3 (Fair. Some noticeable pronunciation errors exist, requiring occasional listener effort for comprehension.)
* 2 (Poor. Pronunciation issues are more pronounced, demanding listener effort for understanding.)
* 1 (Extremely Poor. Very few words are recognizable.)

Fluency (1-5)
* 5 (Fluent without noticeable pauses or stammering.)
* 4 (Mostly Fluent with occasional minor pauses, repetition, and stammering.)
* 3 (Moderately Fluent with noticeable hesitations, but the communication is still coherent.)
* 2 (Not Very Fluent. Speech is characterized by intermittent disruptions and pauses. There are moments of hesitation that may affect the overall flow.)
* 1 (Not Fluent. Communication is challenging, and the speaker may struggle to convey thoughts smoothly.)

Prosody (1-5)
* 5 (Excellent Prosody, expressive and well-modulated tone, enhancing communication with effective pitch and rhythm variations.)
* 4 (Good Prosody, prosodic elements are generally well-controlled, with some variations in pitch and rhythm.)
* 3 (Moderate Prosody, there are noticeable variations in pitch and rhythm, but they may not always align perfectly with the content.)
* 2 (Limited Prosody, speech may sound somewhat monotonous, impacting the overall expressiveness of the communication.)
* 1 (Poor Prosody, speech sound robotic or monotonous.)

**Word level**

* 4 (Good)
* 3 Stress errors (the stress is incorrectly placed on different syllables)
* 3 (Somewhat inaccurate but quite understandable)
* 2 (Challenging to recognize but understandable)
* 1 (Cannot recognize the word)
