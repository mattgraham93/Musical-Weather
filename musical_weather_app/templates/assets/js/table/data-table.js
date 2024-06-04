/*------------------------------------------------------------------
* Bootstrap Responsive Web Application Template
* Email: heyalexluna@gmail.com
* Version: 1.1
* Last change: 2020-03-02
* Author: Alexis Luna
* Copyright 2019 Alexis Luna
* Website: https://github.com/mralexisluna/bootstrap-responsive-web-application-template
-------------------------------------------------------------------*/
// Data Table JavaScripts

(function ($) {
    'use strict';
    
    $('#myTable').DataTable({
        'ajax': {
            'url': '/get_songs',
            'dataSrc': 'selected_songs_season'  // or 'selected_songs_weather' depending on which one you want to display
        },
        'columns': [
            { 'data': 'artist' },
            { 'data': 'track' },
            { 'data': 'event' }
        ]
    });

})(jQuery);