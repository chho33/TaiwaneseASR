# taiwanese-asr

## Warning
This repository is only for demo purpose. Since opening microphone on browsers need https, this project support only running locally so far.

## Run without https
Open your chrome and navigate `chrome://flags/#unsafely-treat-insecure-origin-as-secure`. Find and enable the `Insecure origins treated as secure section`, and then add the address you want to ignore the secure policy.
![image](https://github.com/jojotenya/TaiwaneseASR/blob/master/src/assets/enable%20microphone.png)

## Pre-requests
- Download the model: https://drive.google.com/file/d/1VeSKm1F1t4b1eD73UfRBoiGY_Zhv0-9B/view?usp=sharing
- Decompress the model and move it into the right directory: `tar zxf run.tar.gz; mv run ./backend/;`
- install npm: https://www.npmjs.com/get-npm
- install yarn: https://classic.yarnpkg.com/en/docs/install
- `npm install -g @vue/cli`
- `yarn install`
- `vue add axios`
- `pip install -r requirements.txt`

## Frontend
```
cd src;
yarn serve
```

## Backend
```
cd backend;
# run with cpu
python server.py
# run with gpu
python server.py --config ./available_models/asr.conf.gpu.json
```

![image](https://github.com/jojotenya/TaiwaneseASR/blob/master/src/assets/frontpage.png)
