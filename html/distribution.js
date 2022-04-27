window.onload = main;

const base_path = '../output/'

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

async function create_obj_table() {
    const tabbody = document.getElementById('ObjTab')
    fetch(base_path + '/object_distrib/content.json')
        .then(resp => resp.json())
        .then(df => {
            for (const obj in df) {
                    let path = base_path + `/object_distrib/${obj}.png`

                    const img = document.createElement('img');
                    img.src = path;
                    img.height = 512;

                    tr = createElementWith('tr', [
                        createElementWith('td', [createElementText('label', `${obj}`)]),
                        createElementWith('td', [img]),
                    ]);
                    tabbody.appendChild(tr);
                }
        });
}


async function create_verb_table() {
    const tabbody = document.getElementById('VerbTab')
    fetch(base_path + '/verb_distrib/content.json')
        .then(resp => resp.json())
        .then(df => {
            for (const verb in df) {
                    let path = base_path + `/verb_distrib/${verb}.png`
                    const img = document.createElement('img');
                    img.src = path;
                    img.height = 512;
                    tr = createElementWith('tr', [
                        createElementWith('td', [createElementText('label', `${verb}`)]),
                        createElementWith('td', [img]),
                    ]);
                    tabbody.appendChild(tr);
                }
        });
}


async function main() {
    await create_obj_table();
    await create_verb_table();
}


// Too many Details
// function main() {
//     const tabbody = document.getElementById('ObjTab')

//     fetch(base_path + '/object_distrib/content.json')
//         .then(resp => resp.json())
//         .then(df => {
//             for (const obj in df) {
//                 for (const verb_i in df[obj]) {
//                     const verb = df[obj][verb_i]
//                     let path = base_path + `/object_distrib/${obj}/${verb}.png`

//                     const img = document.createElement('img');
//                     img.src = path;
//                     img.height = 512;

//                     tr = createElementWith('tr', [
//                         createElementWith('td', [createElementText('label', `${obj}`)]),
//                         createElementWith('td', [createElementText('label', `${verb}`)]),
//                         img,
//                     ]);
//                     tabbody.appendChild(tr);

//                 }
//             }

//         });
// }