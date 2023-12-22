import datetime as dt

localinfra = [
    0.0,
    31536000.0,
    63158400.0,
    94694400.0,
    126230400.0,
    157766400.0,
    189388800.0,
    220924800.0,
    252460800.0,
    283996800.0,
    315619200.0,
    347155200.0,
    378691200.0,
    410227200.0,
    441849600.0,
    473385600.0,
    504921600.0,
    536457600.0,
    568080000.0,
    599616000.0,
    631152000.0,
    662688000.0,
    694310400.0,
    725846400.0,
    757382400.0,
    788918400.0,
    820540800.0,
    852076800.0,
    883612800.0,
    915148800.0,
    946771200.0,
    978307200.0,
]
investment = [
    126144000,
    157680000,
    189216000,
    220752000,
    252288000,
    283824000,
    346896000,
    536112000,
]
noise = [
    0.0,
    31536000.0,
    63158400.0,
    94694400.0,
    126230400.0,
    157766400.0,
    189388800.0,
    220924800.0,
    252460800.0,
    283996800.0,
    315619200.0,
    347155200.0,
    378691200.0,
    410227200.0,
    441849600.0,
    473385600.0,
    504921600.0,
    536457600.0,
    568080000.0,
    599616000.0,
    631152000.0,
    662688000.0,
]

def take_closest(myList, myNumber):
    closest = min(myList, key=lambda x:abs(x-myNumber))
    return closest

investment_rounded = [take_closest(localinfra, x) for x in investment]

def convert_to_datetime(timelist):
    timelist = [element + 1546333200 for element in timelist]
    timelist = [dt.datetime.fromtimestamp(x) for x in timelist]
    return timelist

localinfra = convert_to_datetime(localinfra)
investment = convert_to_datetime(investment)
noise = convert_to_datetime(noise)