"""Deterministic 10 m room with first-person rendered waypoints."""
from PIL import Image, ImageDraw
from random import Random
from pathlib import Path

PLACES = ("A", "B", "C")
VARIANTS = ("stay", "seen", "unseen", "counterfactual")
SIZE = (384,256)


def scenario(index, variant):
    rng = Random(76100 + index)
    old, new, other = rng.sample(PLACES, 3)
    route = rng.sample(PLACES, 3)
    if variant == "counterfactual":
        old, new = new, old
    visits = route * 3 + ["H", "H", "H"]
    history = []
    for tick, place in enumerate(visits):
        position = new if variant == "seen" and tick >= 5 else old
        history.append({"tick":tick,"view":place,"blue":position==place,
                        "red":place in PLACES and place!=position and (index+tick)%3==0})
    target = new if variant in ("seen","unseen") else old
    return {"id":f"{index:04d}-{variant}","index":index,"variant":variant,"initial":old,
            "moved_to":new,"target":target,"history":history}


def frame(root, index, spec):
    view = spec["view"]
    filename=f'{index:04d}-{view}-{int(spec["blue"])}-{int(spec.get("red",False))}.jpg'
    path=root/"frames"/filename
    if path.exists():
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    rng=Random(index+4000+ord(view)*100000)
    image=Image.new("RGB",SIZE)
    d=ImageDraw.Draw(image)
    wall=(217+rng.randrange(20),218+rng.randrange(13),212+rng.randrange(17))
    for y in range(256):
        base=wall if y<113 else (151,125,103)
        d.line((0,y,384,y),fill=tuple(max(0,c-int(y/16)) for c in base))
    for x in range(-384,850,80):
        d.line((192,110,x,256),fill=(127,105,86),width=2)
    for y in (122,140,166,210,248):
        d.line((0,y,384,y),fill=(128,106,88),width=1)
    d.line((0,111,384,111),fill=(112,99,88),width=4)
    for x in (31,330):
        d.rectangle((x,36,x+27,95),fill=(95,112,127),outline=(65,79,89),width=2)
        d.line((x+13,36,x+13,95),fill=(148,166,175),width=3)
    shift=rng.randint(-32,32)
    color={"A":(123,75,43),"B":(96,113,91),"C":(138,111,73),"H":(143,122,94)}[view]
    box=lambda coords: tuple(v+shift if i%2==0 else v for i,v in enumerate(coords))
    if view=="A":
        for x in (100,246):
            d.polygon([(x+shift,117),(x+15+shift,117),(x+14+shift,200),(x+shift,205)],fill=(87,52,34))
        d.polygon([(85+shift,100),(110+shift,73),(260+shift,73),(283+shift,100)],fill=(165,111,67))
        d.rectangle(box((85,100,283,119)),fill=color)
    elif view=="B":
        d.rounded_rectangle(box((79,78,298,162)),radius=13,fill=color,outline=(58,74,51),width=4)
        d.rounded_rectangle(box((78,133,307,177)),radius=11,fill=(119,137,108))
        d.rectangle(box((98,175,288,184)),fill=(75,88,69))
    elif view=="C":
        d.rectangle(box((91,73,280,171)),fill=color,outline=(74,59,40),width=4)
        d.line(box((183,74,183,170)),fill=(85,62,46),width=3)
        d.ellipse(box((165,126,173,134)),fill=(51,49,41))
        d.ellipse(box((192,126,200,134)),fill=(51,49,41))
    else:
        d.rectangle(box((145,64,239,121)),fill=color,outline=(97,82,68),width=3)
    if spec.get("blue") or spec.get("red"):
        x=(184 if spec["blue"] else 240)+shift+rng.randrange(-16,17)
        y=173 if view=="A" else 201
        color=(40,128,245) if spec["blue"] else (223,74,58)
        d.ellipse((x-18,y+9,x+18,y+20),fill=(104,91,81))
        d.ellipse((x-15,y-15,x+15,y+15),fill=color,outline=(35,51,71),width=2)
        d.ellipse((x-9,y-9,x-3,y-3),fill=(160,211,255) if spec["blue"] else (253,184,171))
    d.rectangle((0,229,384,256),fill=(41,51,64))
    d.text((10,237),{"A":"A / TABLE","B":"B / SOFA","C":"C / CABINET","H":"H / HALL"}[view],fill="white")
    tmp=path.with_suffix(".tmp")
    image.save(tmp,format="JPEG",quality=90)
    tmp.replace(path)
    return path
