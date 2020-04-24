# taiwanese-asr

## Warning
This repository is only for demo porpuse. Since opening microphone on browsers need https, this project support only running at local so far.

## Pre-requests
- Download model: https://drive.google.com/file/d/1VeSKm1F1t4b1eD73UfRBoiGY_Zhv0-9B/view?usp=sharing
- Decompress: `tar zxf run.tar.gz;`
- Move run file: `mv run ./backend/;`
- install npm: https://www.npmjs.com/get-npm
- install yarn: https://classic.yarnpkg.com/en/docs/install
- `npm install -g @vue/cli-service-global`
- `yarn install`
- `vue add axios`

## Frontend
```
cd src;
yarn serve
```

## Backend
```
cd backend;
python server.py
```
