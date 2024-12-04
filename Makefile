IPYNBS = $(shell find . -name notes.ipynb)
ZIPS = $(IPYNBS:.ipynb=.zip)
HTMLS = $(IPYNBS:.ipynb=.html)
DIRS = $(dir $(IPYNBS))
FIGDIRS = $(foreach dir,$(DIRS),$(dir)figs)
DATADIRS = $(foreach dir,$(DIRS),$(dir)data)

all: $(FIGDIRS) $(DATADIRS) $(HTMLS) $(ZIPS)

%/notes.html: %/notes.ipynb
	jupyter nbconvert --stdout --to=html $< > $@

%/figs: %
	mkdir -p $@

%/data: %
	mkdir -p $@

%/notes.zip: %/notes.html %/notes.ipynb %/figs %/data
	cd $(dir $@); zip -r $(notdir $@) $(notdir $^)

clean:
	rm -f $(HTMLS) $(ZIPS)
	rmdir $(FIGDIRS) 2>/dev/null || true
	rmdir $(DATADIRS) 2>/dev/null || true
	
.PHONY: all clean
