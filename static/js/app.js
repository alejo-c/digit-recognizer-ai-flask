const getElement = id => document.querySelector(id)

const imageInput = getElement('#image-input')
const predictButton = getElement('#predict-btn')
const clearButton = getElement('#clear-btn')
const digitConfidenceLabel = getElement('#digit-confidence-label')

const defaultLineWidth = 2
let scale = 24
let lineWidth = defaultLineWidth
let canvas = getElement('#image-canvas')
let ctx = canvas.getContext('2d')
let bigPixelMatrix = null
let pixelMatrix = null

const clear = () => {
  digitConfidenceLabel.style.display = 'none'
}

const handleImageUpload = () => {
  clear()
  const file = imageInput.files[0]

  if (file) {
    const image = new Image()
    const reader = new FileReader()

    reader.onload = function(e) {
      image.src = e.target.result

      image.onload = function() {
        const width = image.width
        const height = image.height

        canvas.width = width
        canvas.height = height

        ctx.drawImage(image, 0, 0, width, height)
        updatePixelMatrix()
      }
    }

    reader.readAsDataURL(file);
  }
}

const normalizeData = pixelData => {
  const normalizedData = []
  // Loop through the pixel data and convert it to a matrix of values ranging from 0 to 255
  for (let i = 0; i < pixelData.length; i += 4) {
    const r = pixelData[i]
    const g = pixelData[i + 1]
    const b = pixelData[i + 2]
    const grayscaleValue = (r + g + b) / 3 // Convert to grayscale

    // Normalize the grayscale value to a range from 0 to 255
    const normalizedValue = Math.round((grayscaleValue / 255) * 255)

    // Add the normalized value to the matrix
    normalizedData.push(normalizedValue)
  }
  return normalizedData
}

const array2Matrix = (array, size) => {
  let matrix = []
  let row = []
  for (const i of array) {
    row.push(i)
    if (row.length === size) {
      matrix.push(row)
      row = []
    }
  }
  return matrix
}

const decomposeMatrix = (matrix, blockSize) => {
  const blocks = []
  const step = Math.floor(blockSize)
  for (let i = 0; i < matrix.length; i += step) // x
    for (let j = 0; j < matrix.length; j += step) { // y
      let block = []
      for (let k = i; k < i + step; k++) {
        let row = []
        for (let l = j; l < j + step; l++)
          row.push(matrix[k][l])
        block.push(row)
      }
      blocks.push(block)
    }
  return blocks
}

const blockAverages = blocks => {
  const matrix = []
  let row = []
  for (const block of blocks) {
    let sum = 0
    for (i = 0; i < block.length; i++)
      for (j = 0; j < block.length; j++)
        sum += block[i][j]
    row.push(sum / scale ** 2)
    if (row.length === Math.sqrt(blocks.length)) {
      matrix.push(row)
      row = []
    }
  }
  return matrix
}

const fixData = matrix => {
  const fixedMatrix = []
  for (let i = 0; i < matrix.length; i++) {
    let row = []
    for (let j = 0; j < matrix.length; j++)
      row.push([matrix[i][j] / 255])
    fixedMatrix.push(row)
  }
  return fixedMatrix
}

const updatePixelMatrix = () => {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const pixelData = imageData.data
  const normalizedPixels = normalizeData(pixelData)
  bigPixelMatrix = array2Matrix(normalizedPixels, canvas.width)
  const blocks = decomposeMatrix(bigPixelMatrix, scale)
  pixelMatrix = blockAverages(blocks)
}

const loadModel = async () => await tf.loadLayersModel('static/models/digits/digits_model.json')

const drawOutput = output => {
  digitConfidenceLabel.style.display = 'block'
  digitConfidenceLabel.innerText = `Digit: ${output[0][0]}`
}

const predict = () => {
  const digit = fixData(pixelMatrix)
  const input = tf.tensor4d([digit])

  loadModel().then(model => {
    const output = model.predict(input)
    const outputValues = output.arraySync()[0]
    const digitProbabilities = []
    outputValues.forEach((e, i) =>
      digitProbabilities.push([i, +(e * 100).toFixed(4)])
    )
    drawOutput(digitProbabilities.sort((a, b) => b[1] - a[1]))
  })
}

window.addEventListener('DOMContentLoaded', clear)

predictButton.addEventListener('click', predict)
clearButton.addEventListener('click', clear)
imageInput.addEventListener('change', handleImageUpload)
