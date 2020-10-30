function lodsetup() {
	$(".term").mouseover(function(e) {
							 $(".picker.hidden",this).fadeIn(200);
						 });
	$(".term").mouseout(function(e) {
							 $(".picker.hidden",this).fadeOut(200);
						 });
}
