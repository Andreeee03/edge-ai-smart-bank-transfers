import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const EdgeAiApp());
}

class EdgeAiApp extends StatelessWidget {
  const EdgeAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Smart Bank Transfer',
      theme: ThemeData(useMaterial3: true),
      home: const TransferPage(),
    );
  }
}

enum TextAssistMode { completion, normalization }

class TransferPage extends StatefulWidget {
  const TransferPage({super.key});

  @override
  State<TransferPage> createState() => _TransferPageState();
}

class _TransferPageState extends State<TransferPage> {
  static const platform = MethodChannel('edge_ai/native');

  final _beneficiaryController = TextEditingController();
  final _ibanController = TextEditingController();

  final _amountController = TextEditingController();
  final _currencyController = TextEditingController(text: 'EUR');

  final _categoryController = TextEditingController();
  final _referencePeriodController = TextEditingController();
  final _descriptionController = TextEditingController();

  final _suggestion1Controller = TextEditingController();
  final _suggestion2Controller = TextEditingController();

  final _finalDescriptionController = TextEditingController();

  TextAssistMode _textAssistMode = TextAssistMode.completion;

  bool _busy = false;
  bool _aiReady = false;
  bool _aiInitializing = false;
  Future<void>? _aiInitFuture;

  bool _useCalendarContext = false;
  bool _calendarLoading = false;

  List<Map<String, dynamic>> _calendarEvents = [];
  Map<String, dynamic>? _selectedCalendarEvent;

  String _status = 'Ready';

  bool get _hasDescription => _descriptionController.text.trim().isNotEmpty;

  String _clean(String value) {
    return value.trim();
  }

  String _formatAmount(String value) {
    final normalized = value.trim().replaceAll(',', '.');

    final parsed = double.tryParse(normalized);

    if (parsed == null) {
      return normalized;
    }

    return parsed
        .toStringAsFixed(2)
        .replaceFirst(RegExp(r'0+$'), '')
        .replaceFirst(RegExp(r'\.$'), '');
  }

  String _buildPrompt() {
    final beneficiary = _clean(_beneficiaryController.text);

    final amount = _formatAmount(_amountController.text);

    final currency = _clean(_currencyController.text).toUpperCase();

    final category = _clean(_categoryController.text).toUpperCase();

    final referencePeriod = _clean(_referencePeriodController.text);

    final description = _clean(_descriptionController.text);

    final parts = <String>[];

    /*
     * DESCRIPTION EMPTY
     * =================
     * Automatic GENERATION.
     */
    if (description.isEmpty) {
      parts.add(
        'Generate exactly two concise and natural '
        'bank-transfer descriptions using only the '
        'information provided.',
      );

      parts.add(
        'Return two alternative descriptions without '
        'adding unsupported information.',
      );
    }
    /*
     * DESCRIPTION PRESENT + COMPLETE
     * ==============================
     * COMPLETION.
     */
    else if (_textAssistMode == TextAssistMode.completion) {
      parts.add(
        'Complete the following partially written '
        'bank-transfer description.',
      );

      parts.add(
        'Generate exactly two concise and natural '
        'completed alternatives using only the '
        'information provided.',
      );
    }
    /*
     * DESCRIPTION PRESENT + NORMALIZE
     * ===============================
     * NORMALIZATION.
     */
    else {
      parts.add(
        'Normalize the following bank-transfer '
        'description by making it clear, concise '
        'and natural.',
      );

      parts.add(
        'Generate exactly two alternative normalized '
        'descriptions while preserving the original '
        'meaning and without adding unsupported '
        'information.',
      );
    }

    /*
     * Exact field order used by the SFT prompts.
     *
     * IBAN is deliberately NOT included.
     */
    parts.add('Category: $category');
    parts.add('Beneficiary: $beneficiary');
    parts.add('Amount: $amount $currency');

    if (referencePeriod.isNotEmpty) {
      parts.add('Reference period: $referencePeriod');
    }

    if (description.isNotEmpty) {
      if (_textAssistMode == TextAssistMode.completion) {
        parts.add('Partial description: $description');
      } else {
        parts.add('Original description: $description');
      }
    }

    /*
     * Exact inference boundary used in training:
     *
     * prompt.rstrip("\r\n") + "\n\n"
     */
    final prompt = parts.join('\n').replaceFirst(RegExp(r'[\r\n]+$'), '');

    return '$prompt\n\n';
  }

  String? _validateForm() {
    if (_beneficiaryController.text.trim().isEmpty) {
      return 'Insert the beneficiary.';
    }

    if (_amountController.text.trim().isEmpty) {
      return 'Insert the amount.';
    }

    if (_currencyController.text.trim().isEmpty) {
      return 'Insert the currency.';
    }

    if (_categoryController.text.trim().isEmpty) {
      return 'Insert the operation category.';
    }

    return null;
  }

  String _calendarEventKey(Map<String, dynamic> event) {
    return '${event['id']}_${event['startMillis']}';
  }

  String _formatCalendarDate(dynamic value) {
    final millis = value is int ? value : int.tryParse(value.toString());

    if (millis == null) {
      return 'Unknown date';
    }

    final date = DateTime.fromMillisecondsSinceEpoch(millis).toLocal();

    String twoDigits(int value) => value.toString().padLeft(2, '0');

    return '${date.year}-'
        '${twoDigits(date.month)}-'
        '${twoDigits(date.day)}';
  }

  String _calendarEventLabel(Map<String, dynamic> event) {
    final title = (event['title'] ?? 'Untitled event').toString();

    final date = _formatCalendarDate(event['startMillis']);

    return '$title ? $date';
  }

  Future<void> _setCalendarContext(bool enabled) async {
    if (!enabled) {
      setState(() {
        _useCalendarContext = false;
        _calendarEvents = [];
        _selectedCalendarEvent = null;
        _status = 'Calendar context disabled';
      });

      return;
    }

    setState(() {
      _calendarLoading = true;
      _status = 'Requesting calendar access...';
    });

    try {
      var granted =
          await platform.invokeMethod<bool>('hasCalendarPermission') ?? false;

      if (!granted) {
        granted =
            await platform.invokeMethod<bool>('requestCalendarPermission') ??
            false;
      }

      if (!granted) {
        if (!mounted) {
          return;
        }

        setState(() {
          _useCalendarContext = false;
          _calendarEvents = [];
          _selectedCalendarEvent = null;

          _status =
              'Calendar access denied. '
              'AI generation is still available.';
        });

        return;
      }

      final now = DateTime.now();

      final fromMillis = now
          .subtract(const Duration(days: 30))
          .millisecondsSinceEpoch;

      final toMillis = now
          .add(const Duration(days: 365))
          .millisecondsSinceEpoch;

      final raw =
          await platform.invokeMethod<List<dynamic>>('getCalendarEvents', {
            'fromMillis': fromMillis,
            'toMillis': toMillis,
          }) ??
          <dynamic>[];

      final events = raw
          .whereType<Map>()
          .map((event) => Map<String, dynamic>.from(event))
          .where((event) => (event['title'] ?? '').toString().trim().isNotEmpty)
          .toList();

      if (!mounted) {
        return;
      }

      setState(() {
        _useCalendarContext = true;
        _calendarEvents = events;

        _selectedCalendarEvent = events.length == 1 ? events.first : null;

        _status = events.isEmpty
            ? 'Calendar enabled, but no events were found.'
            : 'Calendar ready. Select an event.';
      });
    } on PlatformException catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        _useCalendarContext = false;
        _calendarEvents = [];
        _selectedCalendarEvent = null;

        _status =
            'Calendar unavailable. '
            'AI generation is still available.';
      });

      debugPrint('Calendar error: ${e.code}');
    } finally {
      if (mounted) {
        setState(() {
          _calendarLoading = false;
        });
      }
    }
  }

  Future<void> _initializeAi() async {
    if (_aiReady) {
      return;
    }

    setState(() {
      _aiInitializing = true;
      _status = 'Loading local AI model...';
    });

    try {
      await platform.invokeMethod<String>('loadModel');

      if (mounted) {
        setState(() {
          _status = 'Preparing local inference...';
        });
      }

      await platform.invokeMethod<String>('createContext');

      if (!mounted) {
        return;
      }

      setState(() {
        _aiReady = true;
        _status = 'Local AI ready';
      });
    } on PlatformException catch (e) {
      if (!mounted) {
        return;
      }

      setState(() {
        _aiReady = false;
        _status = 'AI unavailable. Manual description is still available.';
      });

      debugPrint('AI initialization error: ${e.code}');
    } finally {
      if (mounted) {
        setState(() {
          _aiInitializing = false;
        });
      }
    }
  }

  Future<void> _ensureAiReady() async {
    if (_aiReady) {
      return;
    }

    _aiInitFuture ??= _initializeAi();

    await _aiInitFuture;

    if (!_aiReady) {
      throw PlatformException(
        code: 'AI_NOT_READY',
        message: 'Local AI model is not available.',
      );
    }
  }

  String _extractModelText(String raw) {
    const marker = 'Output:\n';

    final index = raw.indexOf(marker);

    if (index >= 0) {
      return raw.substring(index + marker.length).trim();
    }

    return raw.trim();
  }

  List<String> _parseSuggestions(String output) {
    String? first;
    String? second;

    for (final line in output.split('\n')) {
      final trimmed = line.trim();

      final firstMatch = RegExp(r'^1\.\s*(.+)$').firstMatch(trimmed);

      final secondMatch = RegExp(r'^2\.\s*(.+)$').firstMatch(trimmed);

      if (firstMatch != null) {
        first = firstMatch.group(1)?.trim();
      }

      if (secondMatch != null) {
        second = secondMatch.group(1)?.trim();
      }
    }

    return [
      if (first != null && first.isNotEmpty) first,
      if (second != null && second.isNotEmpty) second,
    ];
  }

  Future<void> _generate() async {
    final error = _validateForm();

    if (error != null) {
      setState(() {
        _status = error;
      });

      return;
    }

    setState(() {
      _busy = true;

      _suggestion1Controller.clear();
      _suggestion2Controller.clear();

      _status = 'Preparing generation...';
    });

    try {
      await _ensureAiReady();

      final prompt = _buildPrompt();

      setState(() {
        _status = 'Generating locally on this device...';
      });

      final result = await platform.invokeMethod<String>('generate', {
        'prompt': prompt,
      });

      final text = _extractModelText(result ?? '');

      final suggestions = _parseSuggestions(text);

      if (suggestions.length < 2) {
        setState(() {
          _status = 'The AI did not return two valid suggestions.';
        });

        return;
      }

      _suggestion1Controller.text = suggestions[0];

      _suggestion2Controller.text = suggestions[1];

      setState(() {
        _status = 'Two suggestions generated locally';
      });
    } on PlatformException catch (e) {
      setState(() {
        _status =
            'AI unavailable. You can still enter the description manually.';
      });

      debugPrint('Local inference error: ${e.code}');
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  void _useSuggestion(TextEditingController controller) {
    _finalDescriptionController.text = controller.text.trim();

    setState(() {
      _status = 'AI suggestion selected. You can still edit it.';
    });
  }

  void _rejectSuggestions() {
    _suggestion1Controller.clear();
    _suggestion2Controller.clear();

    setState(() {
      _status = 'Suggestions rejected. Manual entry is still available.';
    });
  }

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _aiInitFuture ??= _initializeAi();
    });
  }

  @override
  void dispose() {
    _beneficiaryController.dispose();
    _ibanController.dispose();

    _amountController.dispose();
    _currencyController.dispose();

    _categoryController.dispose();
    _referencePeriodController.dispose();
    _descriptionController.dispose();

    _suggestion1Controller.dispose();
    _suggestion2Controller.dispose();

    _finalDescriptionController.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final hasSuggestions =
        _suggestion1Controller.text.isNotEmpty ||
        _suggestion2Controller.text.isNotEmpty;

    return Scaffold(
      appBar: AppBar(title: const Text('Smart Bank Transfer')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    const Icon(Icons.smartphone),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'On-device AI â€¢ '
                        'Transfer data is processed locally',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            /*
             * 1. BENEFICIARY
             */
            TextField(
              controller: _beneficiaryController,
              decoration: const InputDecoration(
                labelText: 'Beneficiary',
                border: OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 16),

            /*
             * 2. IBAN
             */
            TextField(
              controller: _ibanController,
              decoration: const InputDecoration(
                labelText: 'IBAN',
                helperText: 'Not used by the AI model',
                border: OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 16),

            /*
             * 3. AMOUNT + CURRENCY
             */
            Row(
              children: [
                Expanded(
                  flex: 2,
                  child: TextField(
                    controller: _amountController,
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                    ),
                    decoration: const InputDecoration(
                      labelText: 'Amount',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),

                const SizedBox(width: 12),

                Expanded(
                  child: TextField(
                    controller: _currencyController,
                    decoration: const InputDecoration(
                      labelText: 'Currency',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            /*
             * 4. CATEGORY
             */
            TextField(
              controller: _categoryController,
              decoration: const InputDecoration(
                labelText: 'Category',
                border: OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 16),

            /*
             * 5. REFERENCE PERIOD
             */
            TextField(
              controller: _referencePeriodController,
              decoration: const InputDecoration(
                labelText: 'Reference period (optional)',
                border: OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 16),

            /*
             * 6. DESCRIPTION
             *
             * Empty = Generation
             * Text  = Completion / Normalization
             */
            TextField(
              controller: _descriptionController,
              maxLines: 3,
              onChanged: (_) {
                setState(() {});
              },
              decoration: const InputDecoration(
                labelText: 'Description (optional)',
                helperText: 'Leave empty to generate a new description',
                border: OutlineInputBorder(),
              ),
            ),

            /*
             * OPTIONAL CALENDAR CONTEXT
             *
             * Available only for Generation.
             */
            if (!_hasDescription) ...[
              const SizedBox(height: 14),

              Card(
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      SwitchListTile(
                        contentPadding: EdgeInsets.zero,
                        title: const Text('Use calendar context'),
                        subtitle: const Text(
                          'Optional. Only the selected event '
                          'is used locally on this device.',
                        ),
                        value: _useCalendarContext,
                        onChanged: (_busy || _calendarLoading)
                            ? null
                            : (value) {
                                _setCalendarContext(value);
                              },
                      ),

                      if (_calendarLoading) ...[
                        const SizedBox(height: 8),
                        const LinearProgressIndicator(),
                        const SizedBox(height: 8),
                        const Text('Loading calendar events...'),
                      ],

                      if (_useCalendarContext && !_calendarLoading) ...[
                        const SizedBox(height: 8),

                        if (_calendarEvents.isEmpty)
                          const Text(
                            'No calendar events found '
                            'in the selected time range.',
                          )
                        else
                          DropdownButtonFormField<String>(
                            isExpanded: true,
                            initialValue: _selectedCalendarEvent == null
                                ? null
                                : _calendarEventKey(_selectedCalendarEvent!),
                            decoration: const InputDecoration(
                              labelText: 'Calendar event',
                              border: OutlineInputBorder(),
                            ),
                            hint: const Text('Select an event'),
                            items: _calendarEvents.map((event) {
                              return DropdownMenuItem<String>(
                                value: _calendarEventKey(event),
                                child: Text(
                                  _calendarEventLabel(event),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              );
                            }).toList(),
                            onChanged: _busy
                                ? null
                                : (key) {
                                    if (key == null) {
                                      return;
                                    }

                                    final event = _calendarEvents.firstWhere(
                                      (item) => _calendarEventKey(item) == key,
                                    );

                                    setState(() {
                                      _selectedCalendarEvent = event;

                                      _status = 'Calendar event selected';
                                    });
                                  },
                          ),

                        if (_selectedCalendarEvent != null) ...[
                          const SizedBox(height: 10),
                          Text(
                            'Selected: '
                            '${_selectedCalendarEvent!['title']}'
                            ' ? '
                            '${_formatCalendarDate(_selectedCalendarEvent!['startMillis'])}',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ],
                    ],
                  ),
                ),
              ),
            ],

            /*
             * Only shown when the user entered
             * an existing description.
             */
            if (_hasDescription) ...[
              const SizedBox(height: 14),

              const Text('What should the AI do with your description?'),

              const SizedBox(height: 8),

              SegmentedButton<TextAssistMode>(
                segments: const [
                  ButtonSegment(
                    value: TextAssistMode.completion,
                    label: Text('Complete'),
                    icon: Icon(Icons.edit),
                  ),
                  ButtonSegment(
                    value: TextAssistMode.normalization,
                    label: Text('Normalize'),
                    icon: Icon(Icons.auto_fix_high),
                  ),
                ],
                selected: {_textAssistMode},
                onSelectionChanged: _busy
                    ? null
                    : (selection) {
                        setState(() {
                          _textAssistMode = selection.first;
                        });
                      },
              ),
            ],

            const SizedBox(height: 22),

            FilledButton.icon(
              onPressed: (_busy || _aiInitializing) ? null : _generate,
              icon: const Icon(Icons.auto_awesome),
              label: Text(
                _busy
                    ? 'Generating...'
                    : hasSuggestions
                    ? 'Regenerate suggestions'
                    : 'Generate suggestions',
              ),
            ),

            if (_busy) ...[
              const SizedBox(height: 14),
              const LinearProgressIndicator(),
            ],

            const SizedBox(height: 18),

            Text(_status, style: Theme.of(context).textTheme.bodySmall),

            if (_suggestion1Controller.text.isNotEmpty) ...[
              const SizedBox(height: 24),

              const Text(
                'AI suggestion 1',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),

              const SizedBox(height: 8),

              TextField(
                controller: _suggestion1Controller,
                maxLines: 2,
                decoration: const InputDecoration(border: OutlineInputBorder()),
              ),

              const SizedBox(height: 8),

              OutlinedButton(
                onPressed: () => _useSuggestion(_suggestion1Controller),
                child: const Text('Use suggestion 1'),
              ),
            ],

            if (_suggestion2Controller.text.isNotEmpty) ...[
              const SizedBox(height: 18),

              const Text(
                'AI suggestion 2',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),

              const SizedBox(height: 8),

              TextField(
                controller: _suggestion2Controller,
                maxLines: 2,
                decoration: const InputDecoration(border: OutlineInputBorder()),
              ),

              const SizedBox(height: 8),

              OutlinedButton(
                onPressed: () => _useSuggestion(_suggestion2Controller),
                child: const Text('Use suggestion 2'),
              ),
            ],

            if (hasSuggestions) ...[
              const SizedBox(height: 10),

              TextButton(
                onPressed: _rejectSuggestions,
                child: const Text('Reject AI suggestions'),
              ),
            ],

            const SizedBox(height: 24),

            const Divider(),

            const SizedBox(height: 16),

            const Text(
              'Final transfer description',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),

            const SizedBox(height: 8),

            TextField(
              controller: _finalDescriptionController,
              maxLines: 3,
              decoration: const InputDecoration(
                hintText: 'Write manually or select an AI suggestion',
                border: OutlineInputBorder(),
              ),
            ),

            const SizedBox(height: 10),

            const Text(
              'The transfer is simulated. '
              'No payment will be executed.',
              style: TextStyle(fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}
