#Growth of the Scientific Boundary

##Summary

This visualization attempts to encode the growth in size and scope of the scientific boundary for multiple disciplines by using the count of published scientific papers to the Arxiv pre-publishing website as a proxy for the "size" of a given scientific boundary. Meta data is collected using [Arxiv's API](http://arxiv.org/help/api/index). The titles of each paper in a given discipline and for a given year are passed into a "Bag of Words" categorization model that sorts each paper based on its abstract into a category, or defines a new category if a new cluster in the model emerges as more paper titles are added as the years progress. Each new category that is created is "born" from a parent category, which is determined as being the existing category that is most-like the newly created category, based on the words in the labels of each category. The bubbles representing each category grow in size based on the fraction of papers within that category, and are colored according to its parent category. The goal is to not only visually represent the growth in scale of knowledge within a scientific discipline, but also the fragmentation of a discipline into greaters numbers of unique specializations.

##Design

###Paper Count Encoding

The original design was to utilize a madelbrot fractal to encode the growing boundry.

![Mandelbrot Fractal](https://github.com/cole-maclean/Udacity-DSNDP6/blob/master/madelbrot_fractal.png)

After obtaining feedback from co-workers on the orignal sketch, many suggested that the complexity of the fractal image took away from the main intent of the visualization.

![Fractal Sketch](https://github.com/cole-maclean/Udacity-DSNDP6/blob/master/fractal_sketch.jpg)

 Using this feedback, the design evolved into a simpliar bubble representation of the dataset. 

![Bubble Sketch](https://github.com/cole-maclean/Udacity-DSNDP6/blob/master/bubble_sketch.jpg)

 This simplier representation of the dataset provides intuitive understanding of the data remaining visually appealing.

###Legend

Originally, each category was uniquelly colored and identified in a legend. After reviewing, it was discovered there were too many categories to effectively distinguish using color and also that coloring each category did not provide additional information as a user could identify each category with a tooltip label. The decision was made to color each category based on its parent category. This providing additional context to the visuallization, providing the ability to quickly identify which categories developed from a given parent category.

##Feedback

###Initial Feedback

Before formally developing the visualization, I received feedback on the conceptual sketch of the mandelbrot fractual representation. Showing this to a couple co-workers, the main feedback received was the apparent complexity of the visuallization and whether a simplier representation would be just as effective. Using this feedback, I did some research of d3.js visualizations looking for inspiration and stumbled upon [this](http://bl.ocks.org/mbostock/4063269) visuallization that seemed suitable for my purposes. Encorporating this feedback into the design led to the current final version of the visualization

###Original Published Visualization Feedback

After building out the conceptual visualization using d3.js and publishing, I received feedback from akmoore on the Udacity forums that included:
1. Adding visualization description detail to the main visualization page
2. Animating the year slider
3. Adding a stemmer to the categorizational model
4. Keeping the parent category legend color consistent as the years progress

I encorprated all but 1 of the reccomendations in the final design. Although the suggestion 4. to keep the parent category legend colors consistent is a valid recommendation, it requires a trade-off in the d3.js implementation of the visualization to either keep the colors consistent, or keep the categories with the same parent category spacially close to each other. After reviewing the suggestion, I decided to keep the categories with the same parent categories spacially close as to allow the user to easily identify categories with the same parent category.

##Resources

Code modified and inspried from:
    http://bl.ocks.org/mbostock/4063269
    http://bl.ocks.org/zanarmstrong/ddff7cd0b1220bc68a58
    http://colorbrewer2.org/
    http://www.d3noob.org/2013/01/adding-tooltips-to-d3js-graph.html