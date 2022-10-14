function simulateClick(tabID) {
  document.getElementById(tabID).click();
}

function predictOnLoad() {
  setTimeout(simulateClick.bind(null, "predict-button"), 500);
}

$(".spinner").hide();

$("#image-selector").change(function () {
  let reader = new FileReader();
  reader.onload = function () {
    let dataURL = reader.result;

    $("#selected-image").attr("src", dataURL);
    $("#displayed-image").attr("src", dataURL);
    $("#prediction-list").empty();

    var canvas = document.getElementById("myCanvas2");
    var ctx = canvas.getContext("2d");
    var img = document.getElementById("color-image");
    ctx.drawImage(img, 0, 0);

    $(".spinner").show();
  };

  let file = $("#image-selector").prop("files")[0];
  reader.readAsDataURL(file);

  setTimeout(simulateClick.bind(null, "predict-button"), 500);
});

let model;
(async function () {
  model = await tf.loadModel(
    "https://ahcstroke.netlify.app/model_3/model.json"
  );
  $("#selected-image").attr(
    "src",
    "https://ahcstroke.netlify.app//assets/ich.jpg"
  );

  $(".progress-bar").hide();

  $(".spinner").show();

  predictOnLoad();
})();

$("#predict-button").click(async function () {
  let image = $("#selected-image").get(0);

  let blank_image = $("#color-image").get(0);

  const verbose = true;

  var orig_image = tf.fromPixels(image);

  var color_image = tf.fromPixels(blank_image);

  orig_image.print(verbose);

  let tensor = tf
    .fromPixels(image)
    .resizeNearestNeighbor([256, 256])
    .toFloat()
    .div(tf.scalar(255.0))
    .expandDims();

  console.log("Input Image shape: ", tensor.shape);

  let predictions = await model.predict(tensor).data();

  var preds = Array.from(predictions);

  var i;
  var num;
  for (i = 0; i < preds.length; i++) {
    num = preds[i];

    if (num < 0.7) {
      preds[i] = 0;
    } else {
      preds[i] = 255;
    }
  }

  pred_tensor = tf.tensor1d(preds, "int32");

  pred_tensor = pred_tensor.reshape([256, 256, 1]);

  pred_tensor = pred_tensor.resizeNearestNeighbor([
    orig_image.shape[0],
    orig_image.shape[1],
  ]);

  rgba_tensor = tf.concat([orig_image, pred_tensor], (axis = -1));
  rgba_tensor = rgba_tensor.resizeNearestNeighbor([250, 250]);
  orig_image = orig_image.resizeNearestNeighbor([250, 250]);
  color_image = color_image.resizeNearestNeighbor([250, 250]);

  var canvas2 = document.getElementById("myCanvas2");
  var canvas3 = document.getElementById("myCanvas3");
  var canvas4 = document.getElementById("myCanvas4");

  tf.toPixels(rgba_tensor, canvas2);
  tf.toPixels(orig_image, canvas3);
  tf.toPixels(color_image, canvas4);

  $(".spinner").hide();
});
