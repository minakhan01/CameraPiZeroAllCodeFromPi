obj-$(CONFIG_AIY_VISION) := aiy-vision.o
ccflags-y := -std=gnu99 -Wno-declaration-after-statement
KVERSION := $(shell uname -r)

all:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) modules

clean:
	$(MAKE) -C /lib/modules/$(KVERSION)/build M=$(PWD) clean

