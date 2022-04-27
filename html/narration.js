window.onload = load_main;

const data_file = 'data/filtered_300.json';
// const data_file = 'data/EPIC100_8k.json';
const epic_root = '/epic_root';
const HEIGHT = 256;
const WIDTH = 456;
const FRAME_WIDTH = 256;

const emap = new Map(); // map from ind to entry element
let prev_hl_ind = 0;  // hightlight index
const frames_arr = new Array();


function render_video(ind) {
    const entry = emap.get(ind).entry;
    const pid = entry.participant_id,
        vid = entry.video_id,
        nid = entry.narration_id,
        st = entry.start_frame,
        ed = entry.stop_frame;
    const video = document.getElementById('video_elem')
    const video_source = document.getElementById('video_source');
    console.log(video_source);
    video_source.setAttribute('src', `../output/narration_grasp/${nid}.mp4`)
    video.load();
    // video.play();

}


function search_vid() {
    const VID_COL_IND = 2;
    const input = document.getElementById("vid_filter");
    const query = input.value;
    const table = document.getElementById("annotTable");
    tr = table.getElementsByTagName("tr");
    for (let i = 1; i < tr.length; i++) {
        const tds = tr[i].getElementsByTagName("td");
        const td = tds[VID_COL_IND];
        let hide = true;
        if (query.length == 0) {
            hide = false;
        } else if (td.innerHTML == query) {
            hide = false;
        }
        if (hide) {
            tr[i].hidden_cnt |= 0x1;
        } else {
            tr[i].hidden_cnt &= ~0x1;
        }
        tr[i].hidden = tr[i].hidden_cnt > 0;
    }
}

function search_frame() {
    const ST_COL_IND = 5, ED_COL_IND = 6;
    const input = document.getElementById("frame_filter");
    const frame = parseInt(input.value);
    const table = document.getElementById("annotTable");
    tr = table.getElementsByTagName("tr");
    for (let i = 1; i < tr.length; i++) {
        const tds = tr[i].getElementsByTagName("td");
        const st_td = tds[ST_COL_IND];
        const ed_td = tds[ED_COL_IND];
        let hide = true;
        if (isNaN(frame)) {
            hide = false;
        } else if ((parseInt(st_td.innerHTML) <= frame) && (frame <= parseInt(ed_td.innerHTML))) {
            hide = false;
        }
        if (hide) {
            tr[i].hidden_cnt |= 0x2;
        } else {
            tr[i].hidden_cnt &= ~0x2;
        }
        tr[i].hidden = tr[i].hidden_cnt > 0;
    }
}

function search_noun() {
    const NOUN_COL_IND = 10;
    const input = document.getElementById("noun_filter");
    const word = input.value; // .toUpperCase();
    let query = null;
    if (word.endsWith(' ')) {
        query = word.slice(0, -1);
    }
    const table = document.getElementById("annotTable");
    tr = table.getElementsByTagName("tr");
    for (let i = 1; i < tr.length; i++) {
        const tds = tr[i].getElementsByTagName("td");
        const td = tds[NOUN_COL_IND];
        let flag = (query != null) ? (td.innerHTML != query) : (td.innerHTML.indexOf(word) < 0);
        if (flag) {
            tr[i].hidden_cnt |= 0x4;
        } else {
            tr[i].hidden_cnt &= ~0x4;
        }
        tr[i].hidden = tr[i].hidden_cnt > 0;
    }
}

function process_key(e) {
    const key = e.key;
    if (key == 40 || key == 'j') {   // hl DOWN
        if (prev_hl_ind != null && prev_hl_ind < emap.size-1) {
            emap.get(prev_hl_ind+1).scrollTo(1000, 100);
            highlight_and_display(prev_hl_ind+1);
        }
    } else if (key == 38 || key == 'k') { // hl UP
        if (prev_hl_ind != null && prev_hl_ind > 0) {
            highlight_and_display(prev_hl_ind-1);
        }
    }
}

function highlight_and_display(hl_ind) {
    emap.get(prev_hl_ind).style.backgroundColor = "gray";
    emap.get(hl_ind).style.backgroundColor = "cyan";
    prev_hl_ind = hl_ind;
    render_video(hl_ind);
}

function load_main() {

    // Create this json using `annot_df.to_json(..., orient='records')
    fetch(data_file)
        .then(resp => resp.json())
        .then(df => {
            df.map( (e, ind) => {
                const entry = createElementWith('tr', [
                    createElementText('td', e.narration_id),
                    createElementText('td', e.participant_id),
                    createElementText('td', e.video_id),
                    createElementText('td', e.narration_timestamp),
                    createElementText('td', e.stop_timestamp),
                    createElementText('td', e.start_frame),
                    createElementText('td', e.stop_frame),
                    createElementText('td', e.narration),
                    createElementText('td', e.verb),
                    createElementText('td', e.verb_class),
                    createElementText('td', e.noun),
                    createElementText('td', e.noun_class),
                    createElementText('td', e.all_nouns),
                    createElementText('td', e.all_noun_classes),
                ], elem => {
                    elem.entry = e;
                    elem.ind = ind;
                    elem.hidden_cnt = 0x0;
                    // e.addEventListener("mouseenter", () => {
                    //     if (!e.selected)
                    //         e.style.backgroundColor = "cyan";
                    // })
                    // e.addEventListener("mouseleave", () => {
                    //     if (!e.selected)
                    //         e.style.backgroundColor = "gray";
                    // })
                    elem.addEventListener("click", () => {
                        highlight_and_display(ind);
                    })
                    return elem;
                });
                const annot_body = document.getElementById('annot_body');
                annot_body.append(entry);
                emap.set(ind, entry);
                return entry;
            })
        }).then( () => {
            // Initialize
            prev_hl_ind = 0;
            highlight_and_display(0);
        })
        .catch(e => console.log(e));

    document.addEventListener('keypress', process_key);

}
