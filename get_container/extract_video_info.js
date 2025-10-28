// extract_video_info.js
const fs = require('fs');
const MP4Box = require('mp4box');

// 存储结果和状态
let resultBoxes = null;
let isReady = false;
let errorMsg = null;

function parseVideo(v_path) {
    if (isReady || errorMsg) return;

    // 检查 v_path
    if (!v_path) {
        errorMsg = "v_path is null or undefined";
        return;
    }
    console.log("v_path:", v_path); // execjs 可能看不到

    try {
        // 检查 fs 模块
        const fs = require('fs');
        if (!fs || !fs.readFileSync) {
            errorMsg = "fs module or readFileSync is not available";
            return;
        }

        console.log("Reading file:", v_path); // execjs 可能看不到
        const arrayBuffer = new Uint8Array(fs.readFileSync(v_path)).buffer;
        console.log("File read success, size:", arrayBuffer.byteLength);

        arrayBuffer.fileStart = 0;

        const mp4file = MP4Box.createFile();
        mp4file.onError = function(e) {
            errorMsg = "MP4Box error: " + (e.message || e);
        };

        mp4file.onReady = function(info) {
            try {
                resultBoxes = mp4file.boxes;
                isReady = true;
                console.log("Parsing success"); // 可能看不到
            } catch (serializeError) {
                errorMsg = "Result serialization failed: " + serializeError.message;
            }
        };

        mp4file.appendBuffer(arrayBuffer);
    } catch (e) {
        errorMsg = "Critical error: " + e.message + " | Stack: " + e.stack;
        console.error(e); // execjs 可能不输出
    }
}

function getMsg(v_path, key, num) {
    if (!isReady && !errorMsg) {
        parseVideo(v_path);
    }

    if (errorMsg) {
        return "error: " + errorMsg;
    }

    if (isReady) {
        if (resultBoxes==null) {
            return "resultBoxesnull";
        }
        if (key == '-1'){
            length = resultBoxes.length;
            re = {}
            for (let i=0; i<length; i++){
                re[i] = Object.keys(resultBoxes[i]);
            }
            return re;
        }else if(key.split('_')[0] == 'boxes'){
            length = resultBoxes.length;
            re = Object.keys(resultBoxes[num]['boxes']);
            const index = Number(key.split('_')[1]);
            let targetKey = re[index];
            let value = resultBoxes[num]['boxes'][targetKey]
            const rest = value && typeof value === 'object' ? (({ samples, ...r }) => r)(value) : value;
            return {
                'value': rest,
                'key': targetKey
            }

        }else if(key.split('_')[0] == 'traks'){
            length = resultBoxes.length;
            re = Object.keys(resultBoxes[num]['traks']);
            const index = Number(key.split('_')[1]);
            let targetKey = re[index];
            let value = resultBoxes[num]['traks'][targetKey]
            const rest = value && typeof value === 'object' ? (({ samples, ...r }) => r)(value) : value;
            return {
                'value': rest,
                'key': targetKey
            }

        }else{
            return resultBoxes[num][key];
        }
        
    }

    if (!isReady){
        return "isReadyFalse";
    }else{
        return "isReadyTrue";
    }
}
// console.log(getMsg('/data/zzw/Full/data2/video/M02_Canon_EOS_SL1/DeviceB/M02_DB_T0098.MOV', 'boxes_4', 1));

