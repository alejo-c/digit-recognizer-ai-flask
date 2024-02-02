const getElement = id => document.querySelector(id)

const scaleInput = getElement('#scale-input')
const lineWidthInput = getElement('#line-width-input')

const scaleLabel = getElement('#scale-label')
const lineWidthLabel = getElement('#line-width-label')

const multipleDrawsCheckbox = getElement('#multiple-draws-checkbox')
const visualize28x28Checkbox = getElement('#visualize-28x28-checkbox')

const clearBtn = getElement('#clear-btn')
const predictBtn = getElement('#predict-btn')
const defaultScaleBtn = getElement('#default-scale-btn')
const defaultLineWidthBtn = getElement('#default-line-width-btn')

const canvas = getElement('#canvas')
const digitConfidenceLabel = getElement('#digit-confidence-label')
const probabilities = getElement('#probabilities')
const loadingArea = getElement('#loading-area')

const ctx = canvas.getContext('2d')
const coord = { x: 0, y: 0 }
const maxScale = 24
const defaultLineWidth = 2

let scale = scaleInput.value = maxScale * .625
let lineWidth = lineWidthInput.value = defaultLineWidth
let isDrawing = false
let itWasDrew = false
let allowsMultipleDraws = false
let canVisualizeAs28x28 = true
let bigPixelMatrix = null
let pixelMatrix = null

const isMobileDevice = () => /Mobi|Android/i.test(navigator.userAgent)
const updateScale = newScale => scale = scaleInput.value = newScale

const clear = () => {
    canvas.width = 28 * scale
    canvas.height = 28 * scale

    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.lineWidth = lineWidth * scale
    ctx.lineCap = 'round'
    ctx.strokeStyle = 'white'

    scaleLabel.innerHTML = scale
    lineWidthLabel.innerHTML = lineWidth
    digitConfidenceLabel.style.display = 'none'
    probabilities.innerHTML = ''

    itWasDrew = false
}

const startDrawing = e => {
    e.preventDefault()

    isDrawing = true
    if (!allowsMultipleDraws)
        clear()

    if (isMobileDevice())
        canvas.addEventListener('touchmove', draw)
    else
        canvas.addEventListener('mousemove', draw)
    reposition(e)

    itWasDrew = true
}

const reposition = e => {
    const { clientX, clientY } = isMobileDevice() ? e.touches[0] : e
    coord.x = clientX - canvas.offsetLeft
    coord.y = clientY - canvas.offsetTop
}

const stopDrawing = () => {
    if (isMobileDevice())
        e.preventDefault()

    if (!isDrawing) return
    isDrawing = false

    if (isMobileDevice())
        canvas.removeEventListener('touchmove', draw)
    else
        canvas.removeEventListener('mousemove', draw)
    updatePixelMatrix()

    if (canVisualizeAs28x28)
        visualizeAs28x28()
    if (!allowsMultipleDraws)
        predict()
}

const draw = e => {
    ctx.beginPath()
    if (!isMobileDevice())
        ctx.moveTo(coord.x, coord.y)
    reposition(e)
    ctx.lineTo(coord.x, coord.y)
    ctx.stroke()
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

const visualizeAs28x28 = () => {
    for (let i = 0; i < pixelMatrix.length; i++)
        for (let j = 0; j < pixelMatrix.length; j++) {
            const pixelValue = Math.floor(pixelMatrix[i][j])
            ctx.fillStyle = `rgb(${pixelValue}, ${pixelValue}, ${pixelValue})`
            ctx.fillRect(j * scale, i * scale, scale, scale)
        }
}

const loadModel = async () => await tf.loadLayersModel('static/models/digits/digits_model.json')

const drawOutput = output => {
    digitConfidenceLabel.style.display = 'block'
    probabilities.innerHTML = ''
    output.forEach(element => {
        if (element[1] > 0)
            probabilities.innerHTML += `<li class='item'>
                <strong class='digit'>
                    ${element[0]} 
                </strong>
                =
                <span class='probability'>
                    ${element[1]}%
                </span>
            </li>`
    })
    loadingArea.style.display = 'none'
}

const predict = () => {
    if (!itWasDrew) return
    loadingArea.style.display = 'flex'

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

scaleInput.addEventListener('input', () => {
    scale = scaleInput.value
    clear()
})
lineWidthInput.addEventListener('input', () => {
    lineWidth = lineWidthInput.value
    clear()
})
multipleDrawsCheckbox.addEventListener('change', () => {
    allowsMultipleDraws = multipleDrawsCheckbox.checked
    clear()
})
visualize28x28Checkbox.addEventListener('change', () => {
    canVisualizeAs28x28 = visualize28x28Checkbox.checked
    clear()
})
defaultScaleBtn.addEventListener('click', () => {
    if (isMobileDevice()) {
        updateScale(Math.floor(window.innerWidth / 28 - 1))
        scaleInput.setAttribute('max', scale)
    } else
        updateScale(maxScale * .625)
    clear()
})
defaultLineWidthBtn.addEventListener('click', () => {
    lineWidth = lineWidthInput.value = defaultLineWidth
    clear()
})

canvas.addEventListener('mousedown', startDrawing)
canvas.addEventListener('mouseup', stopDrawing)
canvas.addEventListener('mouseout', stopDrawing)

canvas.addEventListener('touchstart', startDrawing, false)
canvas.addEventListener('touchend', stopDrawing, false)

clearBtn.addEventListener('click', clear)
predictBtn.addEventListener('click', predict)

document.addEventListener('DOMContentLoaded', () => {
    if (isMobileDevice()) {
        updateScale(Math.floor(window.innerWidth / 28 - 1))
        scaleInput.setAttribute('max', scale)
    }
    scaleInput.setAttribute('max', maxScale)
    clear()
})


window.addEventListener('resize', () => {
    if (isMobileDevice()) {
        updateScale(Math.floor(window.innerWidth / 28) - 1)
        scaleInput.setAttribute('max', scale)
        clear()
    } else if (window.innerWidth < 1100) {
        updateScale(maxScale * .625)
        clear()
    }
})
