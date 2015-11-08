#Growth of the Scientific Boundary

##Summary

This visualization attempts to encode the growth in size and scope of the scientific boundary for multiple disciplines by using the count of published scientific papers to the Arxiv pre-publishing website as a proxy for the "size" of a given scientific boundary. Meta data is collected using [Arxiv's API](http://arxiv.org/help/api/index). The titles of each paper in a given discipline and for a given year are passed into a "Bag of Words" categorization model that sorts each paper based on its abstract into a category, or defines a new category if a new cluster in the model emerges as more paper titles are added as the years progress. Each new category that is created is "born" from a parent category, which is determined as being the existing category that is most-like the newly created category, based on the words in the labels of each category. The bubbles representing each category grow in size based on the fraction of papers within that category, and are colored according to its parent category. The goal is to not only visually represent the growth in scale of knowledge within a scientific discipline, but also the fragmentation of a discipline into greaters numbers of unique specializations.

##Design

![Mandelbrot Fractal](madelbrot_fractal.png)


