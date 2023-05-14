# ChatBot

1. Chatbot is **live** on http://smart-bot.surge.sh/ (use http only and not https) You can use this link to interact with it.

2. If you want to use locally. The python version is in runtime.txt file and all the required dependencies are in requirements.txt file.
You just need to **pip install requirements.txt** file to install all the required dependencies.

3. If you are using conda instead of pip, all the conda packages are present in environment.yml file.

4. If you dont want to interact with the frontend and want to run using command line, just run **python main.py** in the root directory. The main python file is in the root directory as main.py 

5. The app file is in app/server/app.py. I have used flask to run the app, you just have to make one small change at the top
sys.path.insert(0, '/root/ChatBot/helper_codes') at this line, give your system path. and then you can run command "flask run" and the chatbot will be live on your local. If you want to interact with via front-end, open index.html file, and change the url inside the <script> to 127.0.0.1.
