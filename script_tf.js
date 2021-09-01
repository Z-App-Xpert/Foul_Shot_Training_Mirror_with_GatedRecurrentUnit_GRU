"use strict";

const controls = window;
const LandmarkGrid = window.LandmarkGrid;
const drawingUtils = window;
const mpPose = window;
const poseOptions = {
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.4.1624666670/${file}`;
    }
};
// Our input frames will come from here.

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const controlsElement = document.getElementsByClassName('control-panel')[0];
const canvasCtx = canvasElement.getContext('2d');
// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new controls.FPS();
// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
    spinner.style.display = 'none';
};
const landmarkContainer = document.getElementsByClassName('landmark-grid-container')[0];
const grid = new LandmarkGrid(landmarkContainer, {
    connectionColor: 0xCCCCCC,
    definedColors: [{ name: 'LEFT', value: 0xffa500 }, { name: 'RIGHT', value: 0x00ffff }],
    range: 2,
    fitToGrid: true,
    labelSuffix: 'm',
    landmarkSize: 2,
    numCellsPerAxis: 4,
    showHidden: false,
    centered: true,
});


// tensorflow cont variables
var label_array =  ['0.65', '0.75', '0.85', '0.9', '0.95', 'Dribbling', 'Empty', 'Pre-Set', 'SET'];
var x_mean = [ 0.85020781 , 0.49047973 , 0.55461076 , 122.86378239 , 51.6677048 , 154.44922748  ] ;
var x_var = [ 0.06801826 , 0.02439912 , 0.0286541 , 1865.33498448 , 2532.70516074 , 522.4068379  ];

var model_path ;
const model_type = 'gru';

// switch cases for selecting the model path based on the type
// rnn , lstm , gru
switch (model_type) {
    case 'gru' :
        model_path = './models/gru/model.json'; 
        break;
    case 'rnn' :
        model_path = './models/simple_rnn/model.json'; 
        break;
    case 'lstm' :
        model_path = './models/simple_lstm/model.json';
        break;
    default:
        model_path = './models/gru/model.json';
  }

// function for normalize data
function Normalize( x_in , y_in  , visibility , angle , shangle , catangle  ){

    var visibility = (visibility - x_mean[0]) / Math.sqrt( x_var[0] ) ;
    var x = ( x_in - x_mean[1] ) / Math.sqrt( x_var[1] )   ;
    var y = ( y_in - x_mean[2] ) / Math.sqrt( x_var[2] ) ;
    var sew = ( angle - x_mean[3] ) / Math.sqrt( x_var[3] ) ;
    var wsr = ( shangle - x_mean[4] ) / Math.sqrt( x_var[4] ) ;
    var hka = ( catangle - x_mean[5] ) / Math.sqrt( x_var[5] ) ;
  
    return [  visibility , x , y , sew , wsr , hka ];
  
}

// update the input tensor with dynamic updating
function update_input( normalize_data  ){

    model_input[0].shift();
    model_input[0].push( normalize_data  );
  
    return model_input
}

//get the index for the most probabalistic class
function findIndexOfGreatest(array) {
    var greatest;
    var indexOfGreatest;
    for (var i = 0; i < array.length; i++) {
      if (!greatest || array[i] > greatest) {
        greatest = array[i];
        indexOfGreatest = i;
      }
    }
    return indexOfGreatest;
}

// load the model
async function loadModel(path){
    console.log("Model loading in progress from ".concat(path));
    const model =  await tf.loadLayersModel( path);
    console.log("Model Loaded Successfully");
    return model;
}

// initialize the model input instance with all zeros
function zeros(dimensions) {
    var array = [];
    for (var i = 0; i < dimensions[0]; ++i) {
        array.push(dimensions.length == 1 ? 0 : zeros(dimensions.slice(1)));
    }
    return array;
}


// initialize the model input with all zeros and dynamically shift the window by one time step
var model_input = zeros( [1,32,6] );

// load the model from desired type
const model = loadModel(model_path);
console.log("Model Loaded");
let action  ;

function convertTime(time) {
        return time
    // return time.getFullYear() + '-' + ('0' + (time.getMonth() + 1)).slice(-2) + '-' +
    //     ('0' + time.getDate()).slice(-2) + ' ' + time.getHours() + ':' +
    //     ('0' + (time.getMinutes())).slice(-2) + ':' + time.getSeconds();
}

function radians_to_degrees(radians) {
    const pi = Math.PI;
    return radians * (180 / pi);
}

function getAngle(firstPoint, midPoint, lastPoint) {
    console.log('shoulder x is ' + firstPoint.x + 'and y is ' + firstPoint.y)
    let result = radians_to_degrees(Math.atan2(lastPoint.y - midPoint.y, lastPoint.x - midPoint.x) -
        Math.atan2(firstPoint.y - midPoint.y, firstPoint.x - midPoint.x));
    result = Math.abs(result);
    return result > 180 ? (360 - result) : result;
}

function onResults(results) {

    // Hide the spinner.
    document.body.classList.add('loaded');
    // Update the frame rate.
    fpsControl.tick();
    // Draw the overlays.
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    if (!results.poseWorldLandmarks) {
        grid.updateLandmarks([]);
        return;
    }

    /*
    **********************************************************************
    mpPose.POSE_LANDMARKS_LEFT: LEFT side point array
    mpPose.POSE_LANDMARKS_RIGHT: RIGHT side point array
    mpPose.POSE_LANDMARKS_NEUTRAL: NOSE value
    mpPose.POSE_CONNECTIONS: connection information
    **********************************************************************
    */
   

    /* RIGHT side three points index */
    const shoulderIdx = 12;
    const elbowIdx = 14;
    const wristIdx = 16;
    const rfFootIdx = 32;
    const rHipIdx = 24;
    const rKneeIdx = 26
    const rAnkleIdx = 28
    /* RIGHT side three points index */

    drawingUtils.drawConnectors(canvasCtx, results.poseLandmarks, mpPose.POSE_CONNECTIONS, { visibilityMin: 0.65, color: 'white' });
    // console.log('results.poseLandmarks 0 +++', results.poseLandmarks[0]);

    const visibility = results.poseLandmarks[rfFootIdx]['visibility'];
    console.log('visibility is ' + visibility)
    //Begin If Visibility
    if (visibility > 0) 
    {

   
    document.getElementById('visibility').innerHTML = `${visibility}`;
    document.getElementById('x').innerHTML = `${results.poseLandmarks[shoulderIdx]['x']}`;
    document.getElementById('y').innerHTML = `${results.poseLandmarks[rfFootIdx]['y']}`;
    document.getElementById('part').innerHTML = 'Right SEW';
    document.getElementById('2ndpart').innerHTML = 'Right WSR';
    document.getElementById('3rdpart').innerHTML = 'Right HKA';
    const timeNow = new Date().getTime();
    console.log(timeNow)
    document.getElementById('time').innerHTML = `${convertTime(timeNow)}`;

    /* calculate angle */
    const shoulder = { x: results.poseLandmarks[shoulderIdx]['x'], y: results.poseLandmarks[shoulderIdx]['y'] };
    console.log('shoulder x is ' + shoulder.x + 'and y is ' + shoulder.y)
    // document.getElementById('shoulderx').innerHTML = `${shoulder.x}`;
    const elbow = { x: results.poseLandmarks[elbowIdx]['x'], y: results.poseLandmarks[elbowIdx]['y'] };
    console.log('elbow x is ' + elbow.x + 'and y is ' + elbow.y)
    const wrist = { x: results.poseLandmarks[wristIdx]['x'], y: results.poseLandmarks[wristIdx]['y'] };
    console.log('wrist x is ' + wrist.x + 'and y is ' + wrist.y)
        /*Begin calculate shooting angle  */
    console.log('right foot is ' + rfFootIdx)
    const rfFoot = { x: results.poseLandmarks[rfFootIdx]['x'], y: results.poseLandmarks[rfFootIdx]['y'] };
    console.log('Right foot x is ' + rfFoot.x + 'and y is ' + rfFoot.y)
    /*End calculate shooting angle  */
     /*Begin calculate catapult angle  */
    console.log('right hip is ' + rHipIdx)
    const rHip = { x: results.poseLandmarks[rHipIdx]['x'], y: results.poseLandmarks[rHipIdx]['y'] };
    console.log('Right foot x is ' + rHip.x + 'and y is ' + rHip.y)
    
    console.log('right knee is ' + rKneeIdx)
    const rKnee = { x: results.poseLandmarks[rKneeIdx]['x'], y: results.poseLandmarks[rKneeIdx]['y'] };
    console.log('Right foot x is ' + rKnee.x + 'and y is ' + rKnee.y)
    
    console.log('right ankle is ' + rAnkleIdx)
    const rAnkle = { x: results.poseLandmarks[rAnkleIdx]['x'], y: results.poseLandmarks[rAnkleIdx]['y'] };
    console.log('Right foot x is ' + rAnkle.x + 'and y is ' + rAnkle.y)
    /*End calculate catapult angle  */  
    const angle = getAngle(shoulder, elbow, wrist);
    const shangle = getAngle(wrist, shoulder, rfFoot)
    const catangle = getAngle(rHip, rKnee, rAnkle)

    //Predictions
    console.log('SEW angle is ' + angle)
    console.log('WSR angle is ' + shangle)      
    console.log('HKA angle is ' + catangle)     
    console.log('visibility is ' + visibility)

    // const dribbling = ''
    // const strSet = ''
    // const percConfidence = ''
    var msgAction = ''
    /*
    if (visibility >  0) 
    {
        if (shangle < 25.5)
        {
            console.log('Dribbling')
             msgAction = 'Dribbling'
        }
        if (catangle < 143) 
        {
            console.log('SET')
            msgAction = 'SET'
        }
            if ((shangle >130) && (catangle > 160) )
            {
                // If (catangle > 170)
                // {
                console.log('90%')
                msgAction = '90%'
                // }
            }
        
    }
    */
    var normalize_input = Normalize(  results.poseLandmarks[shoulderIdx]['x'] , results.poseLandmarks[rfFootIdx]['y']  , visibility , angle , shangle , catangle  );
    model_input = update_input( normalize_input); // update the model input vector with currect input
    // convert the input array into tf tensor
    var model_input_tensor = tf.tensor( model_input );

    model.then(function (res) {
        const prediction = res.predict(model_input_tensor).dataSync();
        var index_max = findIndexOfGreatest( prediction );
        action =  label_array[index_max] ;
        console.log('label array 5 is  ' + label_array[5]);
    }, function (err) {
        console.log(err);
    });

    // get the predcited class
    msgAction = action ;
    document.getElementById('actionPrediction').innerHTML = msgAction;
    //End Predictions

    document.getElementById('angle').innerHTML = `${angle}`;
    document.getElementById('shangle').innerHTML = `${shangle}`;
    document.getElementById('catangle').innerHTML = `${catangle}`;

    /* calculate angle */
    // console.log('selfie mode is: ' + controlsElement,{selfieMode}
    let recordArray = [{
        'visibility': visibility,
        'x': results.poseLandmarks[shoulderIdx]['x'],
        'y': results.poseLandmarks[rfFootIdx]['y'],
        'part': 'SEW',
        'time': timeNow,
        'angle': angle,
        'shpart':'WSR',
        'shangle': shangle,
        'catangle': catangle,
        'motionAction': msgAction
    }];

    buildTable(recordArray);

    function buildTable(data) {
        let table = document.getElementById('recordTable')
        for (let i = 0; i < data.length; i++) {
            let row = `<tr>
                <td>${data[i].visibility}</td>
                <td>${data[i].x}</td>
                <td>${data[i].y}</td>
                <td>${data[i].angle}</td>
                <td>${data[i].shangle}</td>
                <td>${data[i].catangle}</td>
                <td>${data[i].time}</td>
                <td>${data[i].motionAction}</td>
                </tr>`
            table.innerHTML += row
        }
    }

    drawingUtils.drawLandmarks(canvasCtx, Object.values(mpPose.POSE_LANDMARKS_LEFT)
        .map(index => results.poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' });
    drawingUtils.drawLandmarks(canvasCtx, Object.values(mpPose.POSE_LANDMARKS_RIGHT)
        .map(index => results.poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' });
    drawingUtils.drawLandmarks(canvasCtx, Object.values(mpPose.POSE_LANDMARKS_NEUTRAL)
        .map(index => results.poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'white' });
    canvasCtx.restore();
    grid.updateLandmarks(results.poseWorldLandmarks, mpPose.POSE_CONNECTIONS, [
        { list: Object.values(mpPose.POSE_LANDMARKS_LEFT), color: 'LEFT' },
        { list: Object.values(mpPose.POSE_LANDMARKS_RIGHT), color: 'RIGHT' },
    ]);
}


}  //End If for visibility


const pose = new mpPose.Pose(poseOptions);
pose.onResults(onResults);
// Present a control panel through which the user can manipulate the solution
// options.
new controls
    .ControlPanel(controlsElement, {
        selfieMode: true,
        modelComplexity: 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    })
    .add([
        new controls.StaticText({ title: 'MediaPipe Pose Control' }),
        fpsControl,
        new controls.Toggle({ title: 'Selfie Mode', field: 'selfieMode' }),
        new controls.SourcePicker({
            onSourceChanged: () => {
                // Resets because this model gives better results when reset between
                // source changes.
                pose.reset();
            },
            onFrame: async(input, size) => {
                const aspect = size.height / size.width;
                let width, height;
                if (window.innerWidth > window.innerHeight) {
                    height = window.innerHeight;
                    width = height / aspect;
                } else {
                    width = window.innerWidth;
                    height = width * aspect;
                }
                canvasElement.width = width;
                canvasElement.height = height;
                await pose.send({ image: input });
            },
        }),
        new controls.Slider({
            title: 'Model Complexity',
            field: 'modelComplexity',
            discrete: ['Lite', 'Full', 'Heavy'],
        }),
        new controls.Toggle({ title: 'Smooth Landmarks', field: 'smoothLandmarks' }),
        new controls.Slider({
            title: 'Min Detection Confidence',
            field: 'minDetectionConfidence',
            range: [0, 1],
            step: 0.01
        }),
        new controls.Slider({
            title: 'Min Tracking Confidence',
            field: 'minTrackingConfidence',
            range: [0, 1],
            step: 0.01
        }),
    ])
    .on(x => {
        const options = x;
        videoElement.classList.toggle('selfie', options.selfieMode);
        pose.setOptions(options);
    });
 
