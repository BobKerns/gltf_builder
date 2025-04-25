#!/usr/bin/env node
'use strict';

// Copyright 2025 by Bob Kerns
// Licensed under the MIT license.
// See LICENSE file in the project root for details.
// This file is part of the glTF Builder project.
//

// This file is used to validate glTF files using the glTF validator.
// It is used by the test suite to validate the glTF files generated
// by the glTF Builder. It is not used by the glTF Builder itself.
//
// It can be run standalone from the command line.

const fs = require('fs');
const path = require('path');
const {parseArgs} = require('node:util');
const validator = require('gltf-validator');

// Parse command line arguments
// See https://nodejs.org/api/util.html#util_util_parseargs_options
// for details on the options.
const args = process.argv.slice(2);
const options = {
    out: {
        type: 'string',
        short: 'o',
    },
    ignoredIssues: {
        type: 'string',
        short: 'i',
        default: '',
    },
    onlyIssues: {
        type: 'string',
        short: 'I',
        default: '',
    },
    maxIssues: {
        type: 'string',
        short: 'm',
        default: '100',
    },
    severityOverrides: {
        type: 'string',
        short: 's',
        default: '',
    },
    writeTimestamp: {
        type: 'boolean',
        short: 't',
        default: true,
    },
    format: {
        type: 'string',
        short: 'f',
        default: '',
    },
    help: {
        type: 'boolean',
        short: 'h',
    },
}

const {values, positionals} = parseArgs({
    args,
    options,
    allowPositionals: true,
});
if (positionals.length != 1 || values.help) {
    console.error('Usage: node test/gltf-validator.js' +
        ' [--ignoredIssues <issue1,issue2,...>]' +
        ' [--maxIssues <max issues>]' +
        ' [--severityOverrides <issue1:severity,issue2:severity,...>]' +
        ' [--writeTimestamp <true|false>]' +
        ' [--format <glb|gltf>]' +
        ' [--out <output file>]' +
        ' <input glTF file>');
    process.exit(1);
}

// console.log('Validating: ' + positionals[0], values, positionals);

const input = path.resolve(positionals[0]);
if (!fs.existsSync(input)) {
    console.error('Input file does not exist: ' + input);
    process.exit(1);
}
const asset = fs.readFileSync(positionals[0]);

const externalResourceFunction = (uri) => {
    const resourcePath = path.resolve(path.dirname(input), uri);
    if (!fs.existsSync(resourcePath)) {
        console.error('External resource does not exist: ' + resourcePath);
        return Promise.reject(new Error('External resource does not exist: ' + resourcePath));
    }
    return new Promise((resolve, reject) => {
        fs.readFile(resourcePath, (err, data) => {
            if (err) {
                console.error('Error reading external resource: ' + resourcePath);
                reject(err);
            } else {
                resolve(new Uint8Array(data));
            }
        });
    }
    );
};

// Write the report to the output file or console
// If the output file is not specified, the report is written to the console.
const write_report = (report) => {
    if (values.out) {
        const out = path.resolve(path.dirname(input), values.out);
        fs.writeFileSync(out, JSON.stringify(report, null, 2));
    } else {
        console.log(JSON.stringify(report, null, 2));
    }
};

// Parse the command line arguments
const ignoredIssues = values['ignoredIssues'].split(',')
    .map(x => x.trim())
    .filter(x => x);
const onlyIssues = values['onlyIssues'].split(',')
    .map(x => x.trim())
    .filter(x => x);
if (onlyIssues.length > 0 && ignoredIssues.length > 0) {
    console.error('Cannot use both onlyIssues and ignoredIssues options.');
    process.exit(1);
}
const maxIssues = parseInt(values['maxIssues']);
const severityOverrides = values['severityOverrides'].split(',')
    .map(x => x.trim())
    .filter(x => x)
    .reduce((acc, x) => {
        const [issue, severity] = x.split(':');
        acc[issue] = parseInt(severity);
        return acc;
    }, {});
const writeTimestamp = values['writeTimestamp'];
const format = values['format'] || undefined;

// Validate the asset
validator.validateBytes(new Uint8Array(asset),
                        {
                            ignoredIssues,
                            maxIssues,
                            severityOverrides,
                            writeTimestamp,
                            format,
                            externalResourceFunction,
                        })
    .then(write_report)
    .catch((error) => console.error('Validation failed: ', error));
