
// convert int to 'frame_xxxx'
function pad(num, size) {
    num = num.toString();
    while (num.length < size) num = "0" + num;
    return num;
}

function createElementText(tag, text, func=e=>e) {
    let elem = document.createElement(tag);
    elem.textContent = text;
    return func(elem);
}
function createElementWith(tag, xs, func=e=>e) {
    let elem = document.createElement(tag);
    for (const x of xs) {
        elem.appendChild(x);
    }
    return func(elem);
}
