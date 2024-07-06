# Automatic AI integrated student attendance inspector
##########################################################

# Description
Using flask framework, OpenCV & face_recognition libraries to recognize faces, using local files and telegram open API to automate the process of checking students' attendance in institutions.

# Deployment
1. ## Clone/Fork this repository
2. ## Assets
   - Add as many users as you please in 'assets/users.txt'
   - for each added user, create 'assets/{user_name}.txt' and 'assets/{user_name}.jpg' picture of each individual
   - update the passwords for each user as you decide in 'assets/users.txt'
3. ## Environmental variables
   - ADMIN - your telegram ID
   - BOT_TOKEN - your telegram bot's API token
   - BOT_ID - your telegram bot's ID

# Future Plans
- *Increase the efficiency of the code*
- *Training the model with the current database from many users*
- *Change experimental parameters reflecting on the performance of the code in practice*
