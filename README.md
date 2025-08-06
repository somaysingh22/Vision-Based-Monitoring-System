# Vision-Based-Monitoring-System
Built a surveillsnce system that detects cell phone usage in real time using Deep Learning(CNN) model MobileNet SSD. Implemented messaging and email alerts . Detection of distance as well as sound on capturing and the distance as well as the time is logged in the database.
#  Setup Instructions
1. Twilio Account Setup
To send SMS notifications:
Go to https://www.twilio.com/try-twilio and create a free account.
After verifying your email and phone:
Get your Account SID and Auth Token from the Twilio Console.
Buy a Twilio phone number that supports SMS.
Put it in the code section of #twilio

2. Gmail SMTP Setup for Sending Emails
To send email alerts:
Enable 2-Step Verification on your Google account.
Go to https://myaccount.google.com/apppasswords
Generate a new App Password for "Mail" and copy it.
Put it in the code section of #mail
