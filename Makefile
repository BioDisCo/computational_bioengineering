IPYNBS = $(shell find . -name notes.ipynb)
SITE_IPYNBS = $(shell find . -path "*/sites/*.ipynb")
SITE_HTMLS = $(SITE_IPYNBS:.ipynb=.html)
ZIPS = $(IPYNBS:.ipynb=.zip)
HTMLS = $(IPYNBS:.ipynb=.html)
DIRS = $(dir $(IPYNBS))
FIGDIRS = $(foreach dir,$(DIRS),$(dir)figs)
DATADIRS = $(foreach dir,$(DIRS),$(dir)data)
SITEDIRS = $(foreach dir,$(DIRS),$(dir)sites)

all: $(FIGDIRS) $(DATADIRS) $(SITEDIRS) $(HTMLS) $(SITE_HTMLS) $(ZIPS)

%/notes.html: %/notes.ipynb
	jupyter nbconvert --stdout --to=html --template nbconvert_template $< > $@

$(SITE_HTMLS): %.html: %.ipynb
	jupyter nbconvert --stdout --to=html --template nbconvert_template $< > $@

%/figs: %
	mkdir -p $@

%/data: %
	mkdir -p $@

%/sites: %
	mkdir -p $@

%/notes.zip: %/notes.html %/notes.ipynb %/figs %/data %/sites
	cd $(dir $@); zip -r $(notdir $@) $(notdir $^)

clean:
	rm -f $(HTMLS) $(SITE_HTMLS) $(ZIPS)
	rmdir $(FIGDIRS) 2>/dev/null || true
	rmdir $(DATADIRS) 2>/dev/null || true
	rmdir $(SITEDIRS) 2>/dev/null || true
	
.PHONY: all clean
