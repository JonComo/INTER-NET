var canvas;
var ctx;
var width = 1920;
var height = 1080;
var center = {x: width/2, y: height/2};
var scale = 2;
var mouse = {x: 0, y: 0};
var mouse_down = false;
var objs = [];
var keys = {};
var data_key;
var learning_rate = .1;
var activation = 0;
var weight_mult = .01;

window.onload = function(evt) {
    canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = width/scale;
    canvas.style.height = height/scale;
    document.getElementById("content").appendChild(canvas);

    ctx = canvas.getContext("2d");
    ctx.lineWidth = 2;
    ctx.font = "20px Helvetica";
    ctx.textAlign = "left";
    width = canvas.width;
    height = canvas.height;

    document.getElementById("in").oninput = generate_network;
    document.getElementById("hidden").oninput = generate_network;
    document.getElementById("out").oninput = generate_network;
    document.getElementById("layers").oninput = generate_network;
    document.getElementById("weight").oninput = generate_network;
    document.getElementById("bias").oninput = generate_network;  
    document.getElementById("activation").oninput = get_settings;
    document.getElementById("lr").oninput = get_settings;

    document.getElementById("lr_text").onchange = function() {
        var lr_text = document.getElementById("lr_text");
        learning_rate = Number(lr_text.value);
        document.getElementById("lr").value = learning_rate;
    };

    document.getElementById("clear_data").onclick = function() {
        for (let i = 0; i < objs.length; i++) {
            let obj = objs[i];
            obj.targets = {};
        }
        data_key = '';
        document.getElementById("data_key").innerHTML = "Data key: cleared";        
    };

    document.getElementById("rand_weights").onclick = function() {
        // randomize weights
        for (let i = 0; i < objs.length; i++) {
            let obj = objs[i];
            obj.randomize_weights();
        }
    };

    generate_network();

    window.onmousemove = function(evt) {
        var rect = canvas.getBoundingClientRect();
        mouse.x = evt.clientX - rect.left;
        mouse.y = evt.clientY - rect.top;
        mouse.x *= scale;
        mouse.y *= scale;

        for (let i = 0; i < objs.length; i++) {
            let obj = objs[i];
            obj.onmousemove();
        }
    };

    window.onmousedown = function(evt) {
        mouse_down = true;

        for (let i = 0; i < objs.length; i++) {
            let obj = objs[i];
            obj.onmousedown();
        }
    };

    window.onmouseup = function(evt) {
        mouse_down = false;

        for (let i = 0; i < objs.length; i++) {
            let obj = objs[i];
            obj.onmouseup();
        }
    };

    window.onkeydown = function(evt) {
        let key = evt.key;

        keys[key] = true;
        
        for (let i = 0; i < objs.length; i++) {
            let obj = objs[i];
            obj.onkeydown(key);
        }

        if (key != 't' && 'abcdefghijklmnopqrstuvwxyz'.indexOf(key) != -1) {
            if (data_key != key) {
                data_key = key;
                document.getElementById("data_key").innerHTML = "Data key: " + data_key;
                // set all obs to go to their targets
                for (let i = 0; i < objs.length; i++) {
                    let obj = objs[i];
                    if (obj.targets[data_key]) {
                        obj.value = obj.targets[data_key];
                    }
                }
            } else {
                // save all objs values as targets
                console.log('saving vals: ');
                for (let i = 0; i < objs.length; i++) {
                    let obj = objs[i];
                    if (obj.parents.length == 0 || obj.output) {
                        obj.targets[data_key] = obj.value;
                    }
                }
            }
        }
    };

    window.onkeyup = function(evt) {
        let key = evt.key;
        if (key in keys) {
            delete keys[key];
        }
    }
    
    loop();
};

function loop() {
    ctx.clearRect(0, 0, width, height);
    
    if ('t' in keys) {
        // train on random key
        let keys = Object.keys(objs[0].targets);
        if (keys.length > 0) {

            for (let iter = 0; iter < 10; iter ++) {
                let idx = Math.floor(Math.random() * keys.length);
                data_key = keys[idx];
        
                for (let i = 0; i < objs.length; i++) {
                    let obj = objs[i];
                    if (obj.parents.length == 0) {
                        // input neuron
                        obj.value = obj.targets[data_key];
                    }
                }
        
                for (let i = 0; i < objs.length; i++) {
                    let obj = objs[i];
                    if (obj.output) {
                        obj.calc();
                    }
                }
        
                for (let i = 0; i < objs.length; i++) {
                    let obj = objs[i];
                    if (obj.output) {
                        obj.pull(obj.value-obj.targets[data_key]);
                    }
                }
            }
        }
    }

    for (let i = 0; i < objs.length; i++) {
        let obj = objs[i];
        if (obj.output) {
            obj.calc();
        }
    }

    for (let i = 0; i < objs.length; i++) {
        let obj = objs[i];
        obj.interact();
    }

    for (let i = 0; i < objs.length; i++) {
        let obj = objs[i];
        obj.render_weights();
    }
    
    for (let i = 0; i < objs.length; i++) {
        let obj = objs[i];
        obj.render();
    }
    
    window.requestAnimationFrame(loop);
}

function dist(ax, ay, bx, by) {
    return Math.sqrt((ax - bx)**2 + (ay - by)**2);
}

function act(x) {
    if (activation == 0) {
        // tanh
        return Math.tanh(x);
    } else if (activation == 1) {
        // sigmoid
        return 1/(1 + Math.exp(-x));
    } else if (activation == 2) {
        //relu
        if (x > 0) {
            return x;
        } else {
            return .1*x;
        }
    } else {
        // linear
        return x;
    }
}

function actp(x) {
    if (activation == 0) {
        // tanh prime
        return 1 - Math.tanh(x)**2;
    } else if (activation == 1) {
        // sigmoid prime
        return act(x) * (1 - act(x));
    } else if (activation == 2) {
        //relu
        if (x > 0) {
            return 1;
        } else {
            return 0.1;
        }
    } else {
        // linear
        return 1;
    }
}

function graph(sx, sy, xmin, xmax, fn, vx, vy) {
    ctx.save();
    ctx.fillStyle = "black";
    ctx.strokeStyle = "gray";

    let gx = xmin;
    let gy = fn(gx);
    let scale = 20;

    ctx.beginPath();
    
    for (; gx < xmax; gx += (xmax-xmin)/100) {
        gy = sy -scale * fn(gx);

        if (gx == xmin) {
            ctx.moveTo(sx+gx*scale, gy);
        } else {
            ctx.lineTo(sx+gx*scale, gy);
        }
    }

    ctx.stroke();

    ctx.beginPath();
    ctx.ellipse(sx+vx*scale, sy -scale * vy, 4, 4, 0, 0, Math.PI*2);
    ctx.fill();
    ctx.restore();
}

function Neuron(x, y) {
    this.x = x;
    this.y = y;
    this.sum = 1;   // weighted_sum
    this.value = 1; // act(weighted_sum)
    this.scale = 30;
    this.dragging = false;
    this.output = false;

    // connections
    this.parents = [];
    this.weights = [];

    this.targets = {};

    this.pinned = false;

    this.calc = function() {
        if (this.parents.length == 0) {
            return this.value;
        } else {
            this.sum = 0;
            for (let i = 0; i < this.parents.length; i++) {
                let w = this.weights[i];
                this.sum += this.parents[i].calc() * w;
            }
            this.value = act(this.sum);
            return this.value;
        }
    };

    this.pull = function(dx) {
        if (this.pinned) {
            return;
        }

        dx *= actp(this.sum);

        for (let i = 0; i < this.parents.length; i++) {
            let w = this.weights[i];
            let parent = this.parents[i];

            parent.pull(dx * w);

            this.weights[i] -= dx * parent.value * learning_rate;
        }
    };

    this.set_parents = function(p) {
        this.parents = p;
        // generate weights
        this.randomize_weights();
    };

    this.randomize_weights = function() {
        this.weights = [];
        for (let i = 0; i < this.parents.length; i ++) {
            this.weights.push((Math.random()*2.0 - 1.0) * weight_mult);
        }
    };

    this.onmousedown = function() {
        this.dragged = false;
        if (Math.abs(this.y - mouse.y) < this.scale && Math.abs(this.render_x() - mouse.x) < this.scale) {
            this.dragging = true;
        }
    };

    this.onmouseup = function() {
        if (this.dragging && !this.dragged) {
            if ('Meta' in keys) {
                if (this.parents.length == 0) {
                    this.value = 0;
                } else {
                    for (let i = 0; i < 100; i ++) {
                        this.pull(this.value * .5);
                        this.calc();
                    }
                }
            } else {
                this.pinned = !this.pinned;
            }
        }

        this.dragging = false;
    };

    this.onmousemove = function() {
        this.dragged = true;
    };

    this.onkeydown = function(key) {
        
    };

    this.interact = function() {
        if (this.dragging && mouse_down) {
            if (this.parents.length > 0) {
                let dx = (this.render_x() - mouse.x)/this.scale;
                this.pull(dx);
            } else {
                let dx = (this.render_x() - mouse.x)/this.scale;
                this.value -= dx;
                if (this.value > 1) {
                    this.value = 1;
                } else if (this.value < -1) {
                    this.value = -1;
                }
            }
        }
    };

    this.render_x = function() {
        return this.x + Math.tanh(this.value) * this.scale;
    };

    this.render = function() {

        if (this.dragging && this.parents.length > 0) {
            // draw graph
            graph(120, 120, -5, 5, act, this.sum, this.value);
            //graph(120, 120, -5, 5, actp, this.sum, actp(this.sum));
        }

        // draw slider bg
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = "blue";
        if (this.value > 0) {
            ctx.strokeStyle = "red";
        }
        ctx.lineWidth = this.scale*2-10;
        ctx.lineCap = "round";
        ctx.globalAlpha = .2;
        if (this.dragging) {
            ctx.globalAlpha = .4;
        }
        ctx.moveTo(this.x - this.scale, this.y);
        ctx.lineTo(this.x + this.scale, this.y);
        ctx.stroke();
        ctx.restore();

        ctx.save();
        if (data_key in this.targets) {
            ctx.beginPath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = "black";
            ctx.ellipse(this.x + Math.tanh(this.targets[data_key])*this.scale, this.y, this.scale+4, this.scale+4, 0, 0, Math.PI * 2.0);
            ctx.stroke();
        }

        ctx.beginPath();
        ctx.lineWidth = 4;
        ctx.fillStyle = "white";
        ctx.strokeStyle = "blue";
        if (this.value > 0) {
            ctx.strokeStyle = "red";
        }

        if (this.pinned) {
            ctx.lineWidth = 6;
            ctx.strokeStyle = "gray";
        }

        let nx = this.render_x();

        ctx.ellipse(nx, this.y, this.scale, this.scale, 0, 0, Math.PI * 2.0);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "black";
        ctx.textAlign = "center";
        ctx.fillStyle = "blue";
        if (this.value > 0) {
            ctx.fillStyle = "red";
        }
        ctx.fillText(Math.round(this.value*100)/100, nx, this.y + 8);
        ctx.restore();
    };

    this.render_weights = function() {
        let nx = this.render_x();

        for (let i = 0; i < this.parents.length; i ++) {
            let p = this.parents[i];
            let w = this.weights[i];

            if (Math.abs(w) < .05) {
                continue;
            } else if (w < 0) {
                ctx.strokeStyle = "blue";
                ctx.fillStyle = "blue";
            } else {
                ctx.strokeStyle = "red";
                ctx.fillStyle = "red";
            }

            let px = p.render_x();

            // weight strength
            let highlight = false;
            if (mouse_down) {
                if (dist(nx, this.y, mouse.x, mouse.y) > this.scale && dist(px, p.y, mouse.x, mouse.y) > p.scale) {
                    
                    let cx = (nx + px)/2;
                    let cy = (this.y + p.y)/2;
                    if (dist(nx, this.y, mouse.x, mouse.y) + dist(px, p.y, mouse.x, mouse.y) < dist(nx, this.y, px, p.y) + .1) {
                        if ('Meta' in keys) {
                            this.weights[i] = 0;
                        }

                        ctx.textAlign = "center";
                        ctx.fillStyle = "black";
                        ctx.fillText(Math.round(w*100)/100, mouse.x, mouse.y-10);
                        highlight = true;
                    }
                }
            }
            
            ctx.save();
            ctx.lineWidth = 1.2;
            if (highlight) {
                ctx.lineWidth = 3;
            }
            ctx.globalAlpha = Math.abs(w);
            if (this.dragging) {
                ctx.lineWidth = 3;
            }
            ctx.beginPath();
            ctx.moveTo(px, p.y);
            ctx.lineTo(nx, this.y);
            ctx.stroke();
            ctx.restore();
        }
    };
}

function get_settings() {
    weight_mult = Number(document.getElementById("weight").value);
    activation = Number(document.getElementById("activation").value);

    var label = document.getElementById("activation_label");
    if (activation == 0) {
        label.innerHTML = "Tanh";
    } else if (activation == 1) {
        label.innerHTML = "Sigmoid";
    } else if (activation == 2) {
        label.innerHTML = "Leaky Relu";
    } else if (activation == 3) {
        label.innerHTML = "Linear";
    }

    learning_rate = Number(document.getElementById("lr").value);
    document.getElementById("lr_text").value = learning_rate;
}

function generate_network() {

    var input = Number(document.getElementById("in").value);
    var hidden = Number(document.getElementById("hidden").value);
    var output = Number(document.getElementById("out").value);
    var num_layers = Number(document.getElementById("layers").value);
    var bias = Number(document.getElementById("bias").value);

    get_settings();

    let layers = [input];
    for (let i = 1; i < num_layers; i++) {
        layers.push(hidden);
    }
    layers.push(output);

    objs = [];

    let net = network(layers, bias);
    objs = objs.concat(net);
}

function network(layers, bias) {
    let w = width/2;
    let h = height*3/4;

    let num_layers = layers.length;

    let px;
    let max_layer_num = Math.max.apply(Math, layers) - 1;
    if (max_layer_num == 0) {
        px = 0;
    } else {
        px = w/max_layer_num;
    }
    
    if (isNaN(px)) {
        px = center.x;
    }

    let py = 0;
    if (num_layers > 1) {
        py = h/(num_layers - 1);
    }

    let new_objs = [];
    let last_layer = [];
    for (let i = 0; i < num_layers; i ++) {
        let l = layers[i];
        
        let cur_layer = [];
        let layer_width;
        if (l == 1) {
            layer_width = 0;
        } else {
            layer_width = (l-1) * px;
        }

        for (let j = 0; j < l; j++) {
            let n = new Neuron(center.x - layer_width/2 + j*px, center.y + h/2 - i*py);

            if (i == num_layers-1) {
                n.output = true;
            }

            if (bias) {
                if (i > 0) {
                    if (j < l-1 || i == num_layers-1) {
                        let prev_l = layers[i-1];
                        n.set_parents(last_layer);
                    }
                }
            } else {
                n.set_parents(last_layer);
            }
            
            cur_layer.push(n);
            new_objs.push(n);
        }

        last_layer = cur_layer;
    }

    return new_objs;
}