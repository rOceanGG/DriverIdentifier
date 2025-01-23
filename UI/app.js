Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "http://127.0.0.1:5000/classify-image",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop files here or click to upload",
        autoProcessQueue: false,
        headers: {
            "Access-Control-Allow-Origin": "*"
        }
    });

    
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function (file) {
            let image_data = file.dataURL;
            
            var url = "http://127.0.0.1:5000/classify-image";

            $.post(url, {
                imageData: image_data
            },function(data, status) {
                console.log(data);
                if (!data || data.length==0) {
                    $("#resultHolder").hide();              
                    $("#error").show();
                    return;
                }
            let drivers = ["AlexanderAlbon", "CharlesLeclerc", "CarlosSainz", "EstebanOcon", "FernandoAlonso", "GabrielBortoleto", "GeorgeRussell", "IsackHadjar", "JackDoohan", "KimiAntonelli", "LewisHamilton", "LiamLawson", "LandoNorris", "LanceStroll", "MaxVerstappen", "NicoHulkenberg", "OliverBearman", "OscarPiastri", "PierreGasly", "YukiTsunoda"];
            //data = {"DriverName": "Kimi Antonelli","DriverProbability": 0.7527951123509639}
        let driverName = data.DriverName;
        let driverProbability = data.DriverProbability;

        if (driverName != "Unknown") {
            $("#error").hide();
            $("#resultHolder").show();

            // Display the matching driver card
            $("#resultHolder").html($("[data-player='" + driverName.replace(/ /g, '') + "']").html());

            // Update probability score
            let elementName = "#score" + driverName.replace(/ /g, '');
            $(elementName).html(driverProbability.toFixed(2)); // Show 2 decimal points
        } else {
            $("#resultHolder").hide();
            $("#error").show();
        }

        // Remove the processed file
        dz.removeFile(file);            
        });
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    console.log( "ready!" );
    $("#error").hide();
    $("#resultHolder").hide();

    init();
});