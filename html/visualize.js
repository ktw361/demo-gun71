// var fs = require('fs');
window.onload = main;

const base_path = '../output/image_visualize';

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

function main() {
    grasps = ['Large Diameter',
        'Small Diameter',
        'Medium Wrap',
        'Adducted Thumb',
        'Light Tool',
        'Prismatic 4 Finger',
        'Prismatic 3 Finger',
        'Prismatic 2 Finger',
        'Palmar Pinch',
        'Power Disk',
        'Power Sphere',
        'Precision Disk',
        'Precision Sphere',
        'Tripod',
        'Fixed Hook',
        'Lateral',
        'Index Finger Extension',
        'Extension Type',
        'Distal',
        'Writing Tripod',
        'Tripod Variation',
        'Parallel Extension',
        'Adduction Grip',
        'Tip Pinch',
        'Lateral Tripod',
        'Sphere 4-Finger',
        'Quadpod',
        'Sphere 3-Finger',
        'Stick',
        'Palmar',
        'Ring',
        'Ventral',
        'Inferior Pincer'];

    const tabbody = document.getElementById('Tab')

    let tr = null;
    for (const grasp of grasps) {
        let dir = `${base_path}/${grasp}`;
        // var files = fs.readdirSync('/output/gun/');
        const img1 = document.createElement('img');
        img1.src = `${base_path}/${grasp}/gun/img_0.jpg`;
        img1.height = 256;
        const img2 = document.createElement('img');
        img2.src = `${base_path}/${grasp}/left/img_0.jpg`;
        img2.height = 256;
        const img3 = document.createElement('img');
        img3.src = `${base_path}/${grasp}/right/img_0.jpg`;
        img3.height = 256;

        console.log(dir);
        tr = createElementWith('tr', [
            createElementWith('td', [createElementText('label', `${grasp}`)]),
            createElementWith('td', [img1]),
            createElementWith('td', [img2]),
            createElementWith('td', [img3]),
        ]);
        tabbody.appendChild(tr);
    }
}