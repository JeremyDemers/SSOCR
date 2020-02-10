# SSOCR
Seven Segment Optical Character Recognition

## Algorithm
![solution](images/solution.png)

```
DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 0, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
}
```

Digital recognition of seven-segment digital tube is relatively simple compared to handwritten numeral.

Detect the existence of the corresponding bit, then encode the image, you can accurately identify the number.


## Requirements
* opencv
* numpy

## Setup
```
git clone https://github.com/JeremyDemers/SSOCR.git
python ssocr.py images/test1.bmp -s
```

## Results
![test1.bmp](images/test1.bmp)
![res1.bmp](images/res1.bmp)
![test2.bmp](images/test2.bmp)
![res2.bmp](images/res2.bmp)

```
$ python ssocr.py -i images/test1.bmp
['-', 3, 0, '.', 3, 7]
$ python ssocr.py -i images/test2.bmp -s
[1, 7, 7, '.', 7]
$ python ssocr.py -i images/test3.bmp -s
[0, 7, 8, '.', 3]
$ python ssocr.py -i images/test4.bmp -s
[0, 7, 2, '.', 6]
```

## Acknowledge
[SSOCR](https://www.unix-ag.uni-kl.de/~auerswal/ssocr/)
